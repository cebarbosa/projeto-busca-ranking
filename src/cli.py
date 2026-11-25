from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from .config import load_config
from .data import materialize_processed
from .features import build_ranking_features
from .paths import ARTIFACTS_DIR, PROC_DIR, PROJECT_ROOT, RAW_DIR
from .pipeline import DEFAULT_MODEL_PATH, run_training_pipeline
from .ranking import load_ranker, score_candidates
from .retrieval import BM25Retriever
from .split import materialize_splits
from .validate_query import inspect_query


def _resolve_default_model_path(cfg: dict[str, Any]) -> Path:
    ranking_cfg = cfg.get("ranking") or {}
    active_name = ranking_cfg.get("active_model")
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


def _cmd_prepare(args: argparse.Namespace) -> None:
    cfg = load_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir is not None else RAW_DIR
    proc_dir = Path(args.proc_dir) if args.proc_dir is not None else PROC_DIR

    summary = materialize_processed(
        raw_dir=raw_dir,
        proc_dir=proc_dir,
    )
    print("Processed layer materialized successfully:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    _ = cfg


def _build_product_name_map(products: pd.DataFrame) -> dict[Any, str]:
    """Map product_id to product_name from the products catalog."""
    if "product_id" not in products.columns or "product_name" not in products.columns:
        return {}
    catalog = products[["product_id", "product_name"]].dropna(subset=["product_id"])
    catalog = catalog.drop_duplicates(subset=["product_id"], keep="first")
    return {
        row["product_id"]: str(row["product_name"])
        for row in catalog.to_dict(orient="records")
        if pd.notna(row["product_name"]) and str(row["product_name"]).strip()
    }


def _cmd_search(args: argparse.Namespace) -> None:
    cfg = load_config()
    retrieval_cfg = cfg.get("retrieval") or {}

    max_chars = args.max_length if args.max_length is not None else None
    try:
        validated = inspect_query(args.query, max_chars=max_chars, cfg=cfg)
    except ValueError as exc:
        print(f"Invalid query: {exc}", file=sys.stderr)
        sys.exit(2)

    if validated.was_truncated:
        print(
            f"Note: query truncated from {validated.original_char_count} to "
            f"{validated.final_char_count} chars: '{validated.text}'"
        )

    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else _resolve_default_model_path(cfg)
    )
    if not model_path.exists():
        print(
            f"Model artifact not found at '{model_path}'. Run training or save the model artifact first.",
            file=sys.stderr,
        )
        sys.exit(1)

    proc_path = Path(args.proc_dir) if args.proc_dir is not None else PROC_DIR
    products = pd.read_parquet(proc_path / "products.parquet")
    product_names = _build_product_name_map(products)

    top_k = int(args.top_k) if args.top_k is not None else int(retrieval_cfg.get("top_k", 100))
    limit = int(args.limit) if args.limit is not None else 10

    query_df = pd.DataFrame([{"query_id": 0, "query": validated.text}])
    retriever = BM25Retriever().fit(products)
    candidates = retriever.retrieve(query_df, top_k=top_k)

    if candidates.empty:
        print("No matching candidates found.")
        return

    feat_df = build_ranking_features(
        candidates=candidates,
        queries=query_df,
        products=products,
        labels=None,
    )

    ranker, feat_cols = load_ranker(model_path)
    scored = score_candidates(ranker, feat_df, feat_cols)

    sort_cols = [c for c in ("score", "bm25_score", "product_id") if c in scored.columns]
    ascending = [c == "product_id" for c in sort_cols]
    ranked = scored.sort_values(sort_cols, ascending=ascending, kind="mergesort").head(limit)

    for rank_idx, row in enumerate(ranked.to_dict(orient="records"), start=1):
        pid = row.get("product_id")
        name = product_names.get(pid) or str(pid)
        score_val = float(row.get("score", 0.0))
        print(f"{rank_idx}. [{pid}] {name[:90]} (score={score_val:.4f})")


def _cmd_split(args: argparse.Namespace) -> None:
    cfg = load_config()
    proc_dir = Path(args.proc_dir) if args.proc_dir is not None else PROC_DIR

    splits = materialize_splits(
        proc_dir=proc_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        cfg=cfg,
    )
    print("Splits materialized:")
    print(splits["split"].value_counts().to_dict())


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config()
    proc_dir = Path(args.proc_dir) if args.proc_dir is not None else PROC_DIR
    model_path = Path(args.model_path) if args.model_path is not None else None

    summary = run_training_pipeline(
        proc_dir=proc_dir,
        model_path=model_path,
        top_k=args.top_k,
        cfg=cfg,
    )
    print("Training complete:")
    print(f"  Model saved:         {summary['model_path']}")
    print(f"  Features used:       {summary['num_features']}")
    print(f"  Val retrieval:       {summary['val_retrieval']}")
    print(f"  Val BM25 NDCG@10:    {summary['val_bm25_ranking']['ndcg@10']:.4f}")
    print(f"  Val Ranker NDCG@10:  {summary['val_ranker_ranking']['ndcg@10']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-ranking",
        description="Product search retrieval and ranking pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Materialize cleaned WANDS data from raw to processed layer.",
    )
    prepare_parser.add_argument("--raw-dir", default=None, help="Raw data directory.")
    prepare_parser.add_argument("--proc-dir", default=None, help="Processed data directory.")
    prepare_parser.set_defaults(func=_cmd_prepare)

    search_parser = subparsers.add_parser(
        "search",
        help="Run a product search query.",
    )
    search_parser.add_argument("query", help="Search query string.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Candidate cutoff per query (defaults to config.yaml).",
    )
    search_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max allowed query length (defaults to config.yaml).",
    )
    search_parser.add_argument(
        "--proc-dir",
        default=None,
        help="Processed data directory.",
    )
    search_parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained ranker artifact (defaults to config.yaml).",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of top ranked results to display.",
    )
    search_parser.set_defaults(func=_cmd_search)

    split_parser = subparsers.add_parser(
        "split",
        help="Generate deterministic train/val/test query splits.",
    )
    split_parser.add_argument("--proc-dir", default=None, help="Processed data directory.")
    split_parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Validation query fraction (defaults to config.yaml).",
    )
    split_parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Test query fraction (defaults to config.yaml).",
    )
    split_parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed (defaults to config.yaml).",
    )
    split_parser.set_defaults(func=_cmd_split)

    train_parser = subparsers.add_parser(
        "train",
        help="Run candidate retrieval, feature generation, and ranker training.",
    )
    train_parser.add_argument(
        "--proc-dir",
        default=None,
        help="Processed data directory.",
    )
    train_parser.add_argument(
        "--model-path",
        default=None,
        help="Output path for trained ranker artifact (defaults to config.yaml).",
    )
    train_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Candidate cutoff per query (defaults to config.yaml).",
    )
    train_parser.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()