import pandas as pd
import pytest

from src.evaluation import evaluate_ranking, evaluate_retrieval, map_relevance
from src.features import build_ranking_features
from src.ranking import get_feature_columns, score_candidates, train_ranker
from src.retrieval import BM25Retriever
from src.split import make_query_splits


def _sample_catalog_and_queries():
    products = pd.DataFrame(
        [
            {
                "product_id": 101,
                "product_name": "Solid Oak Dining Table",
                "product_category": "Dining",
                "average_rating": 4.7,
                "rating_count": 120,
                "review_count": 45,
            },
            {
                "product_id": 102,
                "product_name": "Pedestal Bathroom Sink Ceramic",
                "product_category": "Bath",
                "average_rating": 4.5,
                "rating_count": 80,
                "review_count": 20,
            },
            {
                "product_id": 103,
                "product_name": "Closet Door Sliding Mirror Kit",
                "product_category": "Storage",
                "average_rating": 4.2,
                "rating_count": 30,
                "review_count": 10,
            },
            {
                "product_id": 104,
                "product_name": "Oak Coffee Table Modern",
                "product_category": "Living Room",
                "average_rating": 4.6,
                "rating_count": 95,
                "review_count": 35,
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": 1, "query": "oak dining table"},
            {"query_id": 2, "query": "pedestal sink"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"query_id": 1, "product_id": 101, "label": "exact"},
            {"query_id": 1, "product_id": 104, "label": "partial"},
            {"query_id": 2, "product_id": 102, "label": "exact"},
        ]
    )
    labels["relevance"] = map_relevance(labels["label"])
    return products, queries, labels


def test_retrieval_and_ranking_end_to_end():
    products, queries, labels = _sample_catalog_and_queries()

    retriever = BM25Retriever().fit(products)
    cands = retriever.retrieve(queries, top_k=3)
    assert not cands.empty
    assert {"query_id", "product_id", "score"}.issubset(cands.columns)

    rec = evaluate_retrieval(cands, labels, ks=[1, 3])
    assert 0.0 <= rec["recall@1"] <= rec["recall@3"] <= 1.0

    feat_df = build_ranking_features(
        candidates=cands,
        queries=queries,
        products=products,
        labels=labels,
    )
    feat_cols = get_feature_columns(feat_df)
    assert len(feat_cols) > 0

    model, used_cols, _ = train_ranker(feat_df, val_df=feat_df, feature_cols=feat_cols)
    scored = score_candidates(model, feat_df, used_cols)
    assert "score" in scored.columns
    assert scored["score"].notna().all()

    ndcg = evaluate_ranking(scored, ks=[3])
    assert 0.0 <= ndcg["ndcg@3"] <= 1.0


def test_perfect_vs_zero_ndcg():
    ideal = pd.DataFrame(
        [
            {"query_id": 1, "score": 10.0, "relevance": 3},
            {"query_id": 1, "score": 5.0, "relevance": 1},
            {"query_id": 1, "score": 1.0, "relevance": 0},
        ]
    )
    zero = pd.DataFrame(
        [
            {"query_id": 1, "score": 10.0, "relevance": 0},
            {"query_id": 1, "score": 5.0, "relevance": 0},
        ]
    )
    assert evaluate_ranking(ideal, ks=[3])["ndcg@3"] == pytest.approx(1.0)
    assert evaluate_ranking(zero, ks=[3])["ndcg@3"] == pytest.approx(0.0)


def test_retrieval_respects_top_k_and_unseen_query():
    products, _, _ = _sample_catalog_and_queries()
    retriever = BM25Retriever().fit(products)

    unseen_q = pd.DataFrame([{"query_id": 99, "query": "xyznonexistentterm"}])
    cands = retriever.retrieve(unseen_q, top_k=2)

    assert len(cands) <= 2
    assert {"query_id", "product_id", "score"}.issubset(cands.columns)


def test_ndcg_penalizes_reversed_ranking():
    better_order = pd.DataFrame(
        [
            {"query_id": 1, "score": 0.9, "relevance": 2},
            {"query_id": 1, "score": 0.5, "relevance": 1},
            {"query_id": 1, "score": 0.1, "relevance": 0},
        ]
    )
    worse_order = pd.DataFrame(
        [
            {"query_id": 1, "score": 0.9, "relevance": 0},
            {"query_id": 1, "score": 0.5, "relevance": 1},
            {"query_id": 1, "score": 0.1, "relevance": 2},
        ]
    )
    ndcg_better = evaluate_ranking(better_order, ks=[3])["ndcg@3"]
    ndcg_worse = evaluate_ranking(worse_order, ks=[3])["ndcg@3"]
    assert ndcg_better > ndcg_worse


def test_ranking_handles_nan_catalog_features():
    products, queries, labels = _sample_catalog_and_queries()
    products.loc[0, ["average_rating", "rating_count", "review_count"]] = float("nan")

    retriever = BM25Retriever().fit(products)
    cands = retriever.retrieve(queries, top_k=3)
    feat_df = build_ranking_features(cands, queries, products, labels=labels)
    feat_cols = get_feature_columns(feat_df)

    model, used_cols, _ = train_ranker(feat_df, val_df=feat_df, feature_cols=feat_cols)
    scored = score_candidates(model, feat_df, used_cols)
    assert scored["score"].notna().all()


def test_make_query_splits_disjoint_and_deterministic():
    queries = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "query": [f"query {i}" for i in range(16)],
        }
    )

    s1 = make_query_splits(queries, val_size=0.2, test_size=0.2, random_state=42)
    s2 = make_query_splits(queries, val_size=0.2, test_size=0.2, random_state=42)

    assert len(s1) == 15
    assert set(s1["split"]).issubset({"train", "val", "test"})

    train_ids = set(s1.loc[s1["split"] == "train", "query_id"])
    val_ids = set(s1.loc[s1["split"] == "val", "query_id"])
    test_ids = set(s1.loc[s1["split"] == "test", "query_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(range(1, 16))
    pd.testing.assert_frame_equal(s1, s2)