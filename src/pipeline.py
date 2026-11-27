from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.evaluation import evaluate_ranking, evaluate_retrieval, map_relevance
from src.features import build_ranking_features
from src.paths import ARTIFACTS_DIR, PROC_DIR, PROJECT_ROOT
from src.ranking import save_ranker, score_candidates, train_ranker
from src.retrieval import BM25Retriever, DenseEmbeddingRetriever, HybridRetriever


DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "hgb_ranker.joblib"


def _resolve_artifact_path(raw_path: Path | str, default_dir: Path = ARTIFACTS_DIR) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if len(path.parts) > 1 and path.parts[0] == default_dir.name:
        return PROJECT_ROOT / path
    return default_dir / path


def _default_embedding_cache_path(
    dense_model: str,
    retrieval_cfg: dict[str, Any],
) -> Path:
    configured = retrieval_cfg.get("embedding_cache_path") or retrieval_cfg.get("cache_path")
    if configured:
        return _resolve_artifact_path(configured, ARTIFACTS_DIR)
    safe_model_name = dense_model.replace("/", "__")
    return ARTIFACTS_DIR / f"dense_embeddings_{safe_model_name}.npz"


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


def _resolve_selected_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    feature_cols: list[str] | None = None,
    target_col: str = "relevance",
) -> list[str]:
    id_and_target = {"query_id", "product_id", "label", "split", target_col}

    if feature_cols is not None:
        candidates = list(feature_cols)
    else:
        feat_cfg = config.get("features") or {}
        configured = feat_cfg.get("selected_columns")
        if isinstance(configured, list) and configured:
            candidates = list(configured)
        else:
            candidates = [
                c
                for c in df.columns
                if c not in id_and_target and pd.api.types.is_numeric_dtype(df[c])
            ]

    selected: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in df.columns and col not in id_and_target and col not in seen:
            seen.add(col)
            selected.append(col)

    if not selected:
        raise ValueError("No valid selected feature columns found in feature DataFrame.")
    return selected


def _subset_feature_df(
    df: pd.DataFrame,
    selected_cols: list[str],
    target_col: str = "relevance",
) -> pd.DataFrame:
    dedup_df = df.loc[:, ~df.columns.duplicated()].copy()
    keep_cols = list(
        dict.fromkeys(
            [c for c in ("query_id", "product_id") if c in dedup_df.columns]
            + [c for c in selected_cols if c in dedup_df.columns]
            + ([target_col] if target_col in dedup_df.columns else [])
        )
    )
    return dedup_df[keep_cols]


def _build_retriever(
    retrieval_cfg: dict[str, Any],
    bm25_k1: float,
    bm25_b: float,
    retrieval_mode: str | None = None,
    cache_path: Path | str | None = None,
    use_cache: bool = True,
) -> BM25Retriever | DenseEmbeddingRetriever | HybridRetriever:
    mode = str(retrieval_mode or retrieval_cfg.get("mode", "hybrid")).strip().lower()
    dense_model = str(retrieval_cfg.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2"))
    dense_batch_size = int(retrieval_cfg.get("dense_batch_size", 128))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))
    pool_multiplier = int(retrieval_cfg.get("pool_multiplier", 2))
    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else _default_embedding_cache_path(dense_model, retrieval_cfg)
    )

    bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)

    if mode == "bm25":
        return bm25
    if mode == "dense":
        return DenseEmbeddingRetriever(
            model_name=dense_model,
            batch_size=dense_batch_size,
            cache_path=resolved_cache_path,
            use_cache=use_cache,
        )
    if mode == "hybrid":
        dense = DenseEmbeddingRetriever(
            model_name=dense_model,
            batch_size=dense_batch_size,
            cache_path=resolved_cache_path,
            use_cache=use_cache,
        )
        return HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            rrf_k=rrf_k,
            pool_multiplier=pool_multiplier,
        )

    raise ValueError(f"Unsupported retrieval mode: {mode}. Expected 'bm25', 'dense', or 'hybrid'.")


def run_training_pipeline(
    proc_dir: Path | str = PROC_DIR,
    model_path: Path | str | None = None,
    top_k: int | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    model_name: str | None = None,
    retrieval_mode: str | None = None,
    embedding_cache_path: Path | str | None = None,
    use_cache: bool = True,
    feature_cols: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the two-stage retrieval and ranker model from config and save artifact."""
    config = cfg or load_config()
    retrieval_cfg = config.get("retrieval") or {}
    ranking_cfg = config.get("ranking") or {}
    models_cfg = ranking_cfg.get("models") or {}

    resolved_mode = str(retrieval_mode or retrieval_cfg.get("mode", "hybrid")).strip().lower()
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

    retriever = _build_retriever(
        retrieval_cfg=retrieval_cfg,
        bm25_k1=resolved_k1,
        bm25_b=resolved_b,
        retrieval_mode=resolved_mode,
        cache_path=embedding_cache_path,
        use_cache=use_cache,
    ).fit(products)

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

    # Stage-1 baseline score before filtering feature columns
    stage1_score_col = next(
        (col for col in ("bm25_score", "score", "rrf_score") if col in val_feat.columns),
        None,
    )
    if stage1_score_col is not None:
        stage1_val_scored = val_feat[["query_id", stage1_score_col, "relevance"]].rename(
            columns={stage1_score_col: "score"}
        )
        stage1_val_ranking = evaluate_ranking(stage1_val_scored, ks=[10])
    else:
        stage1_val_ranking = {}

    selected_cols = _resolve_selected_features(
        df=train_feat,
        config=config,
        feature_cols=feature_cols,
        target_col="relevance",
    )
    train_feat_sel = _subset_feature_df(train_feat, selected_cols, target_col="relevance")
    val_feat_sel = _subset_feature_df(val_feat, selected_cols, target_col="relevance")

    try:
        trained = train_ranker(
            train_df=train_feat_sel,
            val_df=val_feat_sel,
            feature_cols=selected_cols,
            target_col="relevance",
            model_name=selected_model_name,
            cfg=config,
        )
    except TypeError:
        trained = train_ranker(
            train_df=train_feat_sel,
            val_df=val_feat_sel,
            params=model_params,
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

    val_scored = score_candidates(ranker, val_feat_sel, feat_cols)
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
        "retrieval_mode": resolved_mode,
        "top_k": resolved_top_k,
        "selected_features": feat_cols,
        "num_features": len(feat_cols),
        "val_retrieval": val_retrieval_metrics,
        "val_bm25_ranking": stage1_val_ranking,
        "val_ranker_ranking": ranker_val_ranking,
    }