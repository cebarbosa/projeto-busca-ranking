from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.evaluation import evaluate_ranking, evaluate_retrieval, map_relevance
from src.features import build_ranking_features
from src.paths import ARTIFACTS_DIR, PROC_DIR, PROJECT_ROOT
from src.ranking import save_ranker, score_candidates, train_ranker
from src.retrieval import BM25Retriever


DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "hgb_ranker.joblib"


def _resolve_model_path(
    model_path: Path | str | None,
    model_name: str | None,
    config: dict[str, Any],
) -> Path:
    if model_path is not None:
        return Path(model_path)

    ranking_cfg = config.get("ranking") or {}
    active_name = model_name or ranking_cfg.get("active_model")
    models_cfg = ranking_cfg.get("models") or {}

    if active_name and isinstance(models_cfg.get(active_name), dict):
        artifact_name = models_cfg[active_name].get("artifact_name", f"{active_name}.joblib")
        artifact_dir_raw = ranking_cfg.get("artifact_dir")
        if artifact_dir_raw:
            artifact_dir = Path(artifact_dir_raw)
            if not artifact_dir.is_absolute():
                artifact_dir = PROJECT_ROOT / artifact_dir
        else:
            artifact_dir = ARTIFACTS_DIR
        return artifact_dir / artifact_name

    return DEFAULT_MODEL_PATH


def _resolve_model_path(
    model_path: Path | str | None,
    model_name: str | None,
    config: dict[str, Any],
) -> Path:
    if model_path is not None:
        return Path(model_path)

    ranking_cfg = config.get("ranking") or {}
    active_name = model_name or ranking_cfg.get("active_model")
    models_cfg = ranking_cfg.get("models") or {}

    if active_name and isinstance(models_cfg.get(active_name), dict):
        artifact_name = models_cfg[active_name].get("artifact_name", f"{active_name}.joblib")
        artifact_dir_raw = ranking_cfg.get("artifact_dir")
        if artifact_dir_raw:
            artifact_dir = Path(artifact_dir_raw)
            if not artifact_dir.is_absolute():
                artifact_dir = PROJECT_ROOT / artifact_dir
        else:
            artifact_dir = ARTIFACTS_DIR
        return artifact_dir / artifact_name

    return DEFAULT_MODEL_PATH


def run_training_pipeline(
    proc_dir: Path | str = PROC_DIR,
    model_path: Path | str | None = None,
    top_k: int | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    model_name: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the two-stage retrieval and ranker model from config and save artifact."""
    config = cfg or load_config()
    retrieval_cfg = config.get("retrieval") or {}
    ranking_cfg = config.get("ranking") or {}
    models_cfg = ranking_cfg.get("models") or {}

    resolved_top_k = int(top_k if top_k is not None else retrieval_cfg.get("top_k", 100))
    resolved_k1 = float(bm25_k1 if bm25_k1 is not None else retrieval_cfg.get("bm25_k1", 1.5))
    resolved_b = float(bm25_b if bm25_b is not None else retrieval_cfg.get("bm25_b", 0.75))
    selected_model_name = model_name or ranking_cfg.get("active_model", "hgb_default")

    selected_model_cfg = (
        models_cfg.get(selected_model_name)
        if isinstance(models_cfg, dict)
        else None
    )
    model_params = (
        dict(selected_model_cfg.get("params") or {})
        if isinstance(selected_model_cfg, dict)
        else None
    )

    proc_path = Path(proc_dir)

    products = pd.read_parquet(proc_path / "products.parquet")
    queries = pd.read_parquet(proc_path / "queries.parquet")
    labels = pd.read_parquet(proc_path / "labels.parquet").copy()
    splits = pd.read_parquet(proc_path / "splits.parquet")

    cfg_relevance_map = (config.get("data") or {}).get("relevance_map")
    resolved_relevance_map = (
        {str(k).capitalize(): int(v) for k, v in cfg_relevance_map.items()}
        if isinstance(cfg_relevance_map, dict)
        else None
    )
    labels["relevance"] = map_relevance(labels["label"], mapping=resolved_relevance_map)

    train_qids = set(splits.loc[splits["split"] == "train", "query_id"])
    val_qids = set(splits.loc[splits["split"] == "val", "query_id"])

    train_queries = queries[queries["query_id"].isin(train_qids)]
    val_queries = queries[queries["query_id"].isin(val_qids)]
    train_labels = labels[labels["query_id"].isin(train_qids)]
    val_labels = labels[labels["query_id"].isin(val_qids)]

    retriever = BM25Retriever(k1=resolved_k1, b=resolved_b).fit(products)
    train_candidates = retriever.retrieve(train_queries, top_k=resolved_top_k)
    val_candidates = retriever.retrieve(val_queries, top_k=resolved_top_k)

    eval_ks = sorted({k for k in (20, 50, resolved_top_k) if k <= resolved_top_k})
    val_retrieval_metrics = evaluate_retrieval(val_candidates, val_labels, ks=eval_ks)

    train_feat = build_ranking_features(
        candidates=train_candidates,
        queries=train_queries,
        products=products,
        labels=train_labels,
    )
    val_feat = build_ranking_features(
        candidates=val_candidates,
        queries=val_queries,
        products=products,
        labels=val_labels,
    )

    bm25_val_scored = val_feat[["query_id", "bm25_score", "relevance"]].rename(
        columns={"bm25_score": "score"}
    )
    bm25_val_ranking = evaluate_ranking(bm25_val_scored, ks=[10])

    try:
        trained = train_ranker(
            train_df=train_feat,
            val_df=val_feat,
            params=model_params,
        )
    except TypeError:
        trained = train_ranker(
            train_df=train_feat,
            val_df=val_feat,
        )

    if isinstance(trained, tuple) and len(trained) == 3:
        ranker, feat_cols, ranker_meta = trained
    elif isinstance(trained, tuple) and len(trained) == 2:
        ranker, feat_cols = trained
        ranker_meta = {
            "model_name": selected_model_name,
            "model_type": (selected_model_cfg or {}).get("type"),
            "params": model_params or {},
        }
    else:
        raise ValueError("Unexpected return value from train_ranker")

    val_scored = score_candidates(ranker, val_feat, feat_cols)
    ranker_val_ranking = evaluate_ranking(val_scored, ks=[10])

    output = _resolve_model_path(model_path, selected_model_name, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_ranker(ranker, feat_cols, output, meta=ranker_meta)
    except TypeError:
        save_ranker(ranker, feat_cols, output)  

    return {
        "model_path": str(output),
        "model_name": selected_model_name,
        "top_k": resolved_top_k,
        "num_features": len(feat_cols),
        "val_retrieval": val_retrieval_metrics,
        "val_bm25_ranking": bm25_val_ranking,
        "val_ranker_ranking": ranker_val_ranking,
    }