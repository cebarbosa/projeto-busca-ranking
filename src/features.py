from __future__ import annotations

import re
import numpy as np
import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_set(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap_stats(q_tokens: set[str], text: str) -> tuple[int, float, float]:
    """Return (overlap_count, query_coverage, jaccard_similarity)."""
    if not q_tokens:
        return 0, 0.0, 0.0
    d_tokens = _tokenize_set(text)
    if not d_tokens:
        return 0, 0.0, 0.0
    inter = len(q_tokens & d_tokens)
    union = len(q_tokens | d_tokens)
    coverage = inter / len(q_tokens)
    jaccard = inter / union if union > 0 else 0.0
    return inter, float(coverage), float(jaccard)


def build_ranking_features(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    products: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    only_judged: bool = False,
) -> pd.DataFrame:
    """Build query-product ranking features from retrieval candidates and catalog tables.

    Expects `candidates` with ['query_id', 'product_id', 'score', 'rank'].
    If `labels` is provided (with ['query_id', 'product_id', 'relevance']), attaches
    `relevance` for supervised ranker training/evaluation.
    """
    required_cand = {"query_id", "product_id", "score", "rank"}
    missing_cand = required_cand - set(candidates.columns)
    if missing_cand:
        raise ValueError(f"candidates missing required columns: {sorted(missing_cand)}")

    df = candidates.copy()
    df["query_id"] = df["query_id"].astype("int64")
    df["product_id"] = df["product_id"].astype("int64")
    df["bm25_score"] = df["score"].astype("float64")
    df["retrieval_rank"] = df["rank"].astype("int64")
    df["reciprocal_rank"] = 1.0 / df["retrieval_rank"].clip(lower=1)

    # Join query text
    q_sub = queries[["query_id", "query"]].drop_duplicates(subset=["query_id"]).copy()
    q_sub["query_id"] = q_sub["query_id"].astype("int64")
    q_sub["query"] = q_sub["query"].fillna("").astype(str)
    df = df.merge(q_sub, on="query_id", how="left")
    df["query"] = df["query"].fillna("").astype(str)

    # Join product attributes
    prod_cols = [
        "product_id",
        "product_name",
        "product_class",
        "category_hierarchy",
        "product_features",
        "product_description",
        "average_rating",
        "rating_count",
        "review_count",
    ]
    available_prod_cols = [c for c in prod_cols if c in products.columns]
    p_sub = products[available_prod_cols].drop_duplicates(subset=["product_id"]).copy()
    p_sub["product_id"] = p_sub["product_id"].astype("int64")

    for col in ["product_name", "product_class", "category_hierarchy", "product_features", "product_description"]:
        if col in p_sub.columns:
            p_sub[col] = p_sub[col].fillna("").astype(str)

    for col in ["average_rating", "rating_count", "review_count"]:
        if col in p_sub.columns:
            p_sub[col] = pd.to_numeric(p_sub[col], errors="coerce").astype("float64")

    df = df.merge(p_sub, on="product_id", how="left")

    # Product quality & metadata features
    if "average_rating" in df.columns:
        df["has_rating"] = df["average_rating"].notna().astype("int8")
    if "rating_count" in df.columns:
        df["log1p_rating_count"] = np.log1p(df["rating_count"].clip(lower=0.0))
    if "review_count" in df.columns:
        df["log1p_review_count"] = np.log1p(df["review_count"].clip(lower=0.0))
    if "product_name" in df.columns:
        df["name_char_len"] = df["product_name"].str.len().astype("int32")
    if "product_description" in df.columns:
        df["desc_char_len"] = df["product_description"].str.len().astype("int32")

    # Query-level relative score features
    grp_max_score = df.groupby("query_id")["bm25_score"].transform("max")
    grp_mean_score = df.groupby("query_id")["bm25_score"].transform("mean")
    df["bm25_diff_from_max"] = df["bm25_score"] - grp_max_score
    df["bm25_diff_from_mean"] = df["bm25_score"] - grp_mean_score

    # Lexical overlap features per field
    query_token_sets = {
        str(q): _tokenize_set(str(q)) for q in df["query"].unique()
    }
    q_lower_series = df["query"].str.lower()

    q_len_arr = np.fromiter(
        (len(query_token_sets[q]) for q in df["query"]),
        dtype=np.int32,
        count=len(df),
    )
    df["query_token_count"] = q_len_arr

    for field, prefix in [
        ("product_name", "name"),
        ("product_class", "class"),
        ("category_hierarchy", "cat"),
    ]:
        if field not in df.columns:
            continue
        field_series = df[field].fillna("").astype(str)
        field_lower = field_series.str.lower()

        overlaps = np.empty(len(df), dtype=np.int32)
        coverages = np.empty(len(df), dtype=np.float64)
        jaccards = np.empty(len(df), dtype=np.float64)
        exact_substr = np.empty(len(df), dtype=np.int8)

        for idx, (q_str, q_low, d_str, d_low) in enumerate(
            zip(df["query"], q_lower_series, field_series, field_lower)
        ):
            ov, cov, jac = _overlap_stats(query_token_sets[q_str], d_str)
            overlaps[idx] = ov
            coverages[idx] = cov
            jaccards[idx] = jac
            exact_substr[idx] = 1 if (q_low and q_low in d_low) else 0

        df[f"{prefix}_token_overlap"] = overlaps
        df[f"{prefix}_query_coverage"] = coverages
        df[f"{prefix}_jaccard"] = jaccards
        df[f"{prefix}_exact_query_match"] = exact_substr

    # Optional target attachment for supervised training/evaluation
    if labels is not None:
        if "relevance" not in labels.columns:
            raise ValueError("labels DataFrame must contain a 'relevance' column.")
        lbl_sub = labels[["query_id", "product_id", "relevance"]].drop_duplicates(
            subset=["query_id", "product_id"]
        ).copy()
        lbl_sub["query_id"] = lbl_sub["query_id"].astype("int64")
        lbl_sub["product_id"] = lbl_sub["product_id"].astype("int64")
        lbl_sub["relevance"] = lbl_sub["relevance"].astype("int64")
        lbl_sub["_is_judged"] = 1

        df = df.merge(lbl_sub, on=["query_id", "product_id"], how="left")
        df["_is_judged"] = df["_is_judged"].fillna(0).astype("int8")
        df["relevance"] = df["relevance"].fillna(0).astype("int64")

        if only_judged:
            df = df[df["_is_judged"] == 1].copy()

        df = df.drop(columns=["_is_judged"])

    # Drop raw long text columns from final feature matrix if not needed downstream
    drop_raw_cols = [
        c
        for c in ["query", "product_name", "product_class", "category_hierarchy", "product_features", "product_description"]
        if c in df.columns
    ]
    return df.drop(columns=drop_raw_cols)