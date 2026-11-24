from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import evaluate_ranking, evaluate_retrieval, map_relevance
from src.features import build_ranking_features
from src.paths import ARTIFACTS_DIR, PROC_DIR
from src.ranking import save_ranker, score_candidates, train_ranker
from src.retrieval import BM25Retriever


DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "lgbm_ranker.joblib"


def run_training_pipeline(
    proc_dir: Path | str = PROC_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    top_k: int = 100,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    ranker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run two-stage BM25 retrieval, ranking feature generation, LightGBM training, and artifact saving."""
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

    ranker, feat_cols = train_ranker(
        train_df=train_feat,
        val_df=val_feat,
        params=ranker_params,
    )

    val_scored = score_candidates(ranker, val_feat, feat_cols)
    lgbm_val_ranking = evaluate_ranking(val_scored, ks=[10])

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_ranker(ranker, feat_cols, output)

    return {
        "model_path": str(output),
        "top_k": top_k,
        "num_features": len(feat_cols),
        "val_retrieval": val_retrieval_metrics,
        "val_bm25_ranking": bm25_val_ranking,
        "val_lgbm_ranking": lgbm_val_ranking,
    }