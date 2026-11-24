from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd

from src.evaluation import evaluate_ranking, evaluate_retrieval, map_relevance
from src.features import build_ranking_features
from src.paths import ARTIFACTS_DIR, PROC_DIR
from src.ranking import load_ranker, score_candidates
from src.retrieval import BM25Retriever


DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "lgbm_ranker.joblib"


def _fit_ranker_in_subprocess(
    train_feat_path: Path,
    val_feat_path: Path,
    model_path: Path,
    ranker_params: dict[str, Any] | None = None,
) -> list[str]:
    """Run train_ranker + save_ranker in a fresh Python process and surface child errors."""
    params_json = json.dumps(ranker_params or {})
    script = (
        "import json, sys, pandas as pd; "
        "from src.ranking import train_ranker, save_ranker; "
        "train_df = pd.read_parquet(sys.argv[1]); "
        "val_df = pd.read_parquet(sys.argv[2]); "
        "model_path = sys.argv[3]; "
        "params = json.loads(sys.argv[4]) or None; "
        "ranker, feat_cols = train_ranker(train_df, val_df=val_df, params=params); "
        "save_ranker(ranker, feat_cols, model_path); "
        "print(json.dumps(feat_cols))"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(train_feat_path),
            str(val_feat_path),
            str(model_path),
            params_json,
        ],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess ranker training failed (exit {proc.returncode}).\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

    last_line = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last_line)


def run_training_pipeline(
    proc_dir: Path | str = PROC_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    top_k: int = 100,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    ranker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run end-to-end training pipeline: retrieve candidates, build features, train ranker, save artifact."""
    proc_path = Path(proc_dir)

    products = pd.read_parquet(proc_path / "products.parquet")
    queries = pd.read_parquet(proc_path / "queries.parquet")
    labels = pd.read_parquet(proc_path / "labels.parquet").copy()
    splits = pd.read_parquet(proc_path / "splits.parquet")

    labels["relevance"] = map_relevance(labels["label"])

    train_qids = set(splits.loc[splits["split"] == "train", "query_id"])
    val_qids = set(splits.loc[splits["split"] == "val", "query_id"])

    train_queries = queries[queries["query_id"].isin(train_qids)]
    val_queries = queries[queries["query_id"].isin(val_qids)]

    train_labels = labels[labels["query_id"].isin(train_qids)]
    val_labels = labels[labels["query_id"].isin(val_qids)]

    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b).fit(products)
    train_candidates = retriever.retrieve(train_queries, top_k=top_k)
    val_candidates = retriever.retrieve(val_queries, top_k=top_k)

    eval_ks = sorted({k for k in (20, 50, top_k) if k <= top_k})
    retrieval_metrics = evaluate_retrieval(val_candidates, val_labels, ks=eval_ks)

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

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_pq = tmp_path / "train_feat.parquet"
        val_pq = tmp_path / "val_feat.parquet"
        train_feat.to_parquet(train_pq, index=False)
        val_feat.to_parquet(val_pq, index=False)

        feat_cols = _fit_ranker_in_subprocess(
            train_feat_path=train_pq,
            val_feat_path=val_pq,
            model_path=output,
            ranker_params=ranker_params,
        )

    ranker, loaded_cols = load_ranker(output)
    val_scored = score_candidates(ranker, val_feat, loaded_cols or feat_cols)
    lgbm_metrics = evaluate_ranking(val_scored, ks=[10])

    return {
        "model_path": str(output),
        "top_k": top_k,
        "num_features": len(loaded_cols or feat_cols),
        "val_retrieval": retrieval_metrics,
        "val_bm25_ranking": bm25_val_ranking,
        "val_lgbm_ranking": lgbm_metrics,
    }