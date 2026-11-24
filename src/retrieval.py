from __future__ import annotations

import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+")

_TEXT_COLUMNS = [
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_features",
    "product_description",
]


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def build_product_corpus_text(products: pd.DataFrame) -> pd.Series:
    """Concatenate available product text fields into a single searchable string per product."""
    present_cols = [col for col in _TEXT_COLUMNS if col in products.columns]
    if not present_cols:
        raise ValueError("No expected product text columns found in products DataFrame.")

    parts = [
        products[col].fillna("").astype(str).str.strip()
        for col in present_cols
    ]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " " + part
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


class BM25Retriever:
    """Lexical BM25 baseline retriever over the WANDS product catalog."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.product_ids: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def fit(self, products: pd.DataFrame) -> "BM25Retriever":
        """Index product catalog documents for BM25 scoring."""
        docs = build_product_corpus_text(products)
        tokenized_docs = [tokenize(doc) for doc in docs.tolist()]

        self.product_ids = (
            pd.to_numeric(products["product_id"], errors="raise")
            .to_numpy(dtype=np.int64)
        )
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        return self

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        """Retrieve top_k candidate products per query ordered by descending BM25 score."""
        if self.bm25 is None or self.product_ids is None:
            raise RuntimeError("BM25Retriever must be fitted before calling retrieve().")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        n_items = len(self.product_ids)
        k_eff = min(top_k, n_items)
        rows: list[pd.DataFrame] = []

        for row in queries[["query_id", "query"]].itertuples(index=False):
            qid = int(row.query_id)
            q_tokens = tokenize(str(row.query))
            scores = self.bm25.get_scores(q_tokens)

            if k_eff == n_items:
                top_idx = np.argsort(scores)[::-1]
            else:
                part_idx = np.argpartition(scores, -k_eff)[-k_eff:]
                top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]

            rows.append(
                pd.DataFrame(
                    {
                        "query_id": qid,
                        "product_id": self.product_ids[top_idx],
                        "score": scores[top_idx].astype(np.float64),
                        "rank": np.arange(1, len(top_idx) + 1, dtype=np.int64),
                    }
                )
            )

        if not rows:
            return pd.DataFrame(columns=["query_id", "product_id", "score", "rank"])

        return pd.concat(rows, ignore_index=True)