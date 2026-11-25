from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .config import load_config


RELEVANCE_MAP: dict[str, int] = {
    "Exact": 2,
    "Partial": 1,
    "Irrelevant": 0,
}


def map_relevance(
    labels: pd.Series,
    mapping: dict[str, int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.Series:
    """Map WANDS label strings ('Exact', 'Partial', 'Irrelevant') to numeric grades."""
    if mapping is not None:
        active_map = mapping
    else:
        config = cfg or load_config()
        cfg_map = config.get("data", {}).get("relevance_map")
        if cfg_map:
            # Normalize keys to Title Case ('exact' -> 'Exact') so it works either way
            active_map = {str(k).capitalize(): int(v) for k, v in cfg_map.items()}
        else:
            active_map = RELEVANCE_MAP

    return labels.map(active_map).fillna(0).astype(int)


def recall_at_k_for_query(
    ranked_product_ids: list[int],
    relevant_product_ids: set[int],
    k: int,
) -> float | None:
    """Compute Recall@K for a single query. Returns None if query has no relevant items."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant_product_ids:
        return None
    top_k = ranked_product_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_product_ids)
    return hits / len(relevant_product_ids)


def ndcg_at_k_for_query(
    relevances_in_rank_order: list[float],
    k: int,
) -> float | None:
    """Compute NDCG@K for a single query given gains ordered by predicted rank."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevances_in_rank_order:
        return None

    gains = np.asarray(relevances_in_rank_order[:k], dtype=float)
    ideal_gains = np.sort(np.asarray(relevances_in_rank_order, dtype=float))[::-1][:k]

    idcg = float(np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2))))
    if idcg == 0.0:
        return 0.0

    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    return dcg / idcg


def evaluate_retrieval(
    candidates_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ks: list[int] | tuple[int, ...] = (10, 50, 100),
    min_relevant_grade: int = 1,
) -> dict[str, float]:
    """Evaluate retrieval Recall@K across queries.

    Expects `candidates_df` with columns ['query_id', 'product_id'] ordered by
    retrieval score (best first), and `labels_df` with ['query_id', 'product_id', 'relevance'].
    """
    relevant_pairs = labels_df[labels_df["relevance"] >= min_relevant_grade]
    relevant_by_qid: dict[int, set[int]] = (
        relevant_pairs.groupby("query_id")["product_id"].apply(set).to_dict()
    )

    ranked_by_qid: dict[int, list[int]] = (
        candidates_df.groupby("query_id", sort=False)["product_id"].apply(list).to_dict()
    )

    results: dict[str, float] = {}
    for k in ks:
        scores: list[float] = []
        for qid, rel_pids in relevant_by_qid.items():
            rec = recall_at_k_for_query(ranked_by_qid.get(qid, []), rel_pids, k=k)
            if rec is not None:
                scores.append(rec)
        results[f"recall@{k}"] = float(np.mean(scores)) if scores else 0.0

    return results


def evaluate_ranking(
    scored_df: pd.DataFrame,
    ks: list[int] | tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Evaluate ranking NDCG@K across queries.

    Expects `scored_df` with columns ['query_id', 'score', 'relevance'], where
    higher `score` means higher rank.
    """
    ordered = scored_df.sort_values(["query_id", "score"], ascending=[True, False])

    results: dict[str, float] = {}
    for k in ks:
        scores: list[float] = []
        for _, grp in ordered.groupby("query_id", sort=False):
            val = ndcg_at_k_for_query(grp["relevance"].tolist(), k=k)
            if val is not None:
                scores.append(val)
        results[f"ndcg@{k}"] = float(np.mean(scores)) if scores else 0.0

    return results