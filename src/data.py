from __future__ import annotations

from pathlib import Path
import pandas as pd

from .paths import PROC_DIR, RAW_DIR


REQUIRED_PRODUCT_COLS = {"product_id"}
REQUIRED_QUERY_COLS = {"query_id"}
REQUIRED_LABEL_COLS = {"query_id", "product_id", "label"}

PRODUCT_TEXT_COLS = [
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
]
PRODUCT_NUMERIC_COLS = [
    "rating_count",
    "average_rating",
    "review_count",
]
QUERY_TEXT_COLS = [
    "query",
    "query_class",
]

def load_raw_wands(
        raw_dir: Path = RAW_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega product.csv, query.csv e label.csv separados por tabulação."""
    product_path = raw_dir / "product.csv"
    query_path = raw_dir / "query.csv"
    label_path = raw_dir / "label.csv"

    for path in (product_path, query_path, label_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. "
                "Coloque os arquivos do WANDS antes de rodar a preparação."
            )

    products = pd.read_csv(product_path, sep="\t", low_memory=False)
    queries = pd.read_csv(query_path, sep="\t", low_memory=False)
    labels = pd.read_csv(label_path, sep="\t", low_memory=False)

    missing_p = REQUIRED_PRODUCT_COLS - set(products.columns)
    missing_q = REQUIRED_QUERY_COLS - set(queries.columns)
    missing_l = REQUIRED_LABEL_COLS - set(labels.columns)
    if missing_p or missing_q or missing_l:
        raise ValueError(
            f"Colunas ausentes — products: {missing_p}, queries: {missing_q}, labels: {missing_l}"
        )

    return products, queries, labels


def _clean_text_series(series: pd.Series) -> pd.Series:
    """Normalize missing values and collapse extra whitespace/line breaks."""
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"[\r\n\t]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def clean_wands(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean raw WANDS tables into a consistent processed (proc) layer."""
    p = products.copy()
    q = queries.copy()
    l = labels.copy()

    # Fix raw column name typo
    p = p.rename(columns={"category hierarchy": "category_hierarchy"})

    # Enforce explicit integer IDs across tables for safe joins
    p["product_id"] = pd.to_numeric(p["product_id"], errors="raise").astype("int64")
    l["product_id"] = pd.to_numeric(l["product_id"], errors="raise").astype("int64")

    q["query_id"] = pd.to_numeric(q["query_id"], errors="raise").astype("int64")
    l["query_id"] = pd.to_numeric(l["query_id"], errors="raise").astype("int64")

    # Clean text columns without turning nulls into "nan" tokens
    for col in PRODUCT_TEXT_COLS:
        if col in p.columns:
            p[col] = _clean_text_series(p[col])

    for col in QUERY_TEXT_COLS:
        if col in q.columns:
            q[col] = _clean_text_series(q[col])

    # Clean numeric product attributes safely (keeping NaN for tree/ranking models)
    for col in PRODUCT_NUMERIC_COLS:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce").astype("float64")

    # Standardize labels and deduplicate keys
    if "label" in l.columns:
        l["label"] = _clean_text_series(l["label"]).str.capitalize()

    p = p.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)
    if "query_id" in q.columns:
        q = q.drop_duplicates(subset=["query_id"], keep="first").reset_index(drop=True)
    if {"query_id", "product_id"}.issubset(l.columns):
        l = l.drop_duplicates(subset=["query_id", "product_id"], keep="first").reset_index(drop=True)

    return p, q, l


def materialize_processed(
    raw_dir: Path = RAW_DIR,
    proc_dir: Path = PROC_DIR,
) -> dict[str, int]:
    """Load raw WANDS files, apply basic cleaning, and write processed tables to PROC_DIR."""
    proc_dir.mkdir(parents=True, exist_ok=True)

    products_raw, queries_raw, labels_raw = load_raw_wands(raw_dir)

    products_proc, queries_proc, labels_proc = clean_wands(
        products_raw, queries_raw, labels_raw
    )

    products_proc.to_parquet(proc_dir / "products.parquet", index=False)
    queries_proc.to_parquet(proc_dir / "queries.parquet", index=False)
    labels_proc.to_parquet(proc_dir / "labels.parquet", index=False)

    return {
        "products_raw": len(products_raw),
        "products_proc": len(products_proc),
        "queries_raw": len(queries_raw),
        "queries_proc": len(queries_proc),
        "labels_raw": len(labels_raw),
        "labels_proc": len(labels_proc),
    }