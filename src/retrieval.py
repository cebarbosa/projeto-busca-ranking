from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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


def save_embeddings_cache(
    path: Path | str,
    product_ids: np.ndarray,
    embeddings: np.ndarray,
    model_name: str,
) -> Path:
    """Persist catalog embeddings, product IDs, and model identifier to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        product_ids=np.asarray(product_ids, dtype=np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_name=np.array([model_name]),
    )
    return out


def load_embeddings_cache(
    path: Path | str,
    expected_product_ids: np.ndarray,
    expected_model_name: str,
) -> np.ndarray | None:
    """Load cached catalog embeddings if model name and product IDs match."""
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=False)
    cached_ids = data["product_ids"]
    cached_model = str(data["model_name"][0]) if "model_name" in data else ""
    if cached_model != expected_model_name:
        return None
    if (
        len(cached_ids) != len(expected_product_ids)
        or not np.array_equal(cached_ids, expected_product_ids)
    ):
        return None
    return np.asarray(data["embeddings"], dtype=np.float32)


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


def _rrf_ranks(scores: np.ndarray, k_eff: int) -> np.ndarray:
    """Return 1-based rank for each item index (0 is best)."""
    if k_eff == len(scores):
        order = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -k_eff)[-k_eff:]
        order = part[np.argsort(scores[part])[::-1]]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return ranks


class DenseEmbeddingRetriever:
    """Dense semantic retriever over catalog text using SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
        device: str | None = None,
        cache_path: Path | str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.use_cache = use_cache
        self.model: SentenceTransformer | None = None
        self.product_ids: np.ndarray | None = None
        self.doc_embeddings: np.ndarray | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def save_cache(self, path: Path | str | None = None) -> Path:
        """Save current fitted embeddings to disk."""
        target = Path(path) if path is not None else self.cache_path
        if target is None:
            raise ValueError("No cache_path provided to save_cache().")
        if self.product_ids is None or self.doc_embeddings is None:
            raise RuntimeError("DenseEmbeddingRetriever must be fitted before save_cache().")
        return save_embeddings_cache(
            path=target,
            product_ids=self.product_ids,
            embeddings=self.doc_embeddings,
            model_name=self.model_name,
        )

    def fit(
        self,
        products: pd.DataFrame,
        cache_path: Path | str | None = None,
        use_cache: bool | None = None,
    ) -> "DenseEmbeddingRetriever":
        """Load cached embeddings if valid; otherwise encode and L2-normalize catalog texts."""
        resolved_cache = Path(cache_path) if cache_path is not None else self.cache_path
        resolved_use_cache = self.use_cache if use_cache is None else use_cache

        self.product_ids = (
            pd.to_numeric(products["product_id"], errors="raise")
            .to_numpy(dtype=np.int64)
        )

        if resolved_use_cache and resolved_cache is not None:
            cached = load_embeddings_cache(
                path=resolved_cache,
                expected_product_ids=self.product_ids,
                expected_model_name=self.model_name,
            )
            if cached is not None:
                self.doc_embeddings = cached
                return self

        docs = build_product_corpus_text(products).tolist()
        model = self._ensure_model()
        emb = model.encode(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.doc_embeddings = np.asarray(emb, dtype=np.float32)

        if resolved_use_cache and resolved_cache is not None:
            save_embeddings_cache(
                path=resolved_cache,
                product_ids=self.product_ids,
                embeddings=self.doc_embeddings,
                model_name=self.model_name,
            )

        return self

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        """Retrieve top_k products per query ordered by cosine similarity."""
        if self.doc_embeddings is None or self.product_ids is None:
            raise RuntimeError("DenseEmbeddingRetriever must be fitted before retrieve().")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if queries.empty:
            return pd.DataFrame(columns=["query_id", "product_id", "score", "rank"])

        model = self._ensure_model()
        q_texts = queries["query"].fillna("").astype(str).tolist()
        q_embs = model.encode(
            q_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q_embs = np.asarray(q_embs, dtype=np.float32)

        # Normalized embeddings => dot product == cosine similarity
        sim_matrix = q_embs @ self.doc_embeddings.T
        n_items = len(self.product_ids)
        k_eff = min(top_k, n_items)
        rows: list[pd.DataFrame] = []

        qids = pd.to_numeric(queries["query_id"], errors="raise").to_numpy(dtype=np.int64)
        for i, qid in enumerate(qids):
            scores = sim_matrix[i]
            if k_eff == n_items:
                top_idx = np.argsort(scores)[::-1]
            else:
                part_idx = np.argpartition(scores, -k_eff)[-k_eff:]
                top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]

            rows.append(
                pd.DataFrame(
                    {
                        "query_id": int(qid),
                        "product_id": self.product_ids[top_idx],
                        "score": scores[top_idx].astype(np.float64),
                        "rank": np.arange(1, len(top_idx) + 1, dtype=np.int64),
                    }
                )
            )

        return pd.concat(rows, ignore_index=True)


class HybridRetriever:
    """Combines BM25 lexical retrieval and dense embedding retrieval via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseEmbeddingRetriever | None = None,
        rrf_k: int = 60,
        pool_multiplier: int = 2,
    ) -> None:
        self.bm25 = bm25_retriever or BM25Retriever()
        self.dense = dense_retriever or DenseEmbeddingRetriever()
        self.rrf_k = rrf_k
        self.pool_multiplier = max(1, pool_multiplier)

    def fit(self, products: pd.DataFrame) -> "HybridRetriever":
        self.bm25.fit(products)
        self.dense.fit(products)
        return self

    def retrieve(self, queries: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        if (
            self.bm25.bm25 is None
            or self.dense.doc_embeddings is None
            or self.dense.product_ids is None
        ):
            raise RuntimeError("HybridRetriever must be fitted before calling retrieve().")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if queries.empty:
            return pd.DataFrame(columns=["query_id", "product_id", "score", "rank"])

        model = self.dense._ensure_model()
        q_embs = np.asarray(
            model.encode(
                queries["query"].fillna("").astype(str).tolist(),
                batch_size=self.dense.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        sim_matrix = q_embs @ self.dense.doc_embeddings.T

        n_items = len(self.dense.product_ids)
        pool_k = min(n_items, max(top_k, top_k * self.pool_multiplier))
        k_eff = min(top_k, n_items)
        rows: list[pd.DataFrame] = []

        qids = pd.to_numeric(queries["query_id"], errors="raise").to_numpy(dtype=np.int64)
        for i, (qid, q_text) in enumerate(zip(qids, queries["query"].fillna("").astype(str))):
            bm25_scores = self.bm25.bm25.get_scores(tokenize(q_text))
            dense_scores = sim_matrix[i]

            bm25_part = (
                np.argpartition(bm25_scores, -pool_k)[-pool_k:]
                if pool_k < n_items
                else np.arange(n_items)
            )
            dense_part = (
                np.argpartition(dense_scores, -pool_k)[-pool_k:]
                if pool_k < n_items
                else np.arange(n_items)
            )
            cand_idx = np.union1d(bm25_part, dense_part)

            bm25_order = cand_idx[np.argsort(bm25_scores[cand_idx])[::-1]]
            dense_order = cand_idx[np.argsort(dense_scores[cand_idx])[::-1]]

            rrf_scores = np.zeros(n_items, dtype=np.float64)
            for r_zero, idx in enumerate(bm25_order, start=1):
                rrf_scores[idx] += 1.0 / (self.rrf_k + r_zero)
            for r_zero, idx in enumerate(dense_order, start=1):
                rrf_scores[idx] += 1.0 / (self.rrf_k + r_zero)

            if k_eff == n_items:
                top_idx = np.argsort(rrf_scores)[::-1]
            else:
                part_idx = np.argpartition(rrf_scores, -k_eff)[-k_eff:]
                top_idx = part_idx[np.argsort(rrf_scores[part_idx])[::-1]]

            rows.append(
                pd.DataFrame(
                    {
                        "query_id": int(qid),
                        "product_id": self.dense.product_ids[top_idx],
                        "score": rrf_scores[top_idx],
                        "rank": np.arange(1, len(top_idx) + 1, dtype=np.int64),
                    }
                )
            )

        return pd.concat(rows, ignore_index=True)