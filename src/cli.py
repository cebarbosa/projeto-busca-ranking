from __future__ import annotations

import argparse
from pathlib import Path

from src.data import materialize_processed
from src.split import materialize_splits
from src.paths import ARTIFACTS_DIR, PROC_DIR, RAW_DIR
from src.pipeline import DEFAULT_MODEL_PATH, run_training_pipeline


def _cmd_prepare(args: argparse.Namespace) -> None:
    summary = materialize_processed(
        raw_dir=Path(args.raw_dir),
        proc_dir=Path(args.proc_dir),
    )
    print("Processed layer materialized successfully:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")


def _cmd_search(args: argparse.Namespace) -> None:
    query = (args.query or "").strip()
    if not query:
        print("Error: search query cannot be empty.")
        raise SystemExit(1)
    if len(query) > args.max_length:
        print(f"Error: query exceeds maximum length of {args.max_length} characters.")
        raise SystemExit(1)

    # Placeholder until retrieval + ranking pipeline is wired up
    print(f"Query accepted: {query!r} (top_k={args.top_k})")


def _cmd_split(args: argparse.Namespace) -> None:
    splits = materialize_splits(
        proc_dir=Path(args.proc_dir),
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Splits materialized:")
    print(splits["split"].value_counts().to_dict())

def _cmd_train(args: argparse.Namespace) -> None:
    summary = run_training_pipeline(
        proc_dir=Path(args.proc_dir),
        model_path=Path(args.model_path),
        top_k=args.top_k,
    )
    print("Training complete:")
    print(f"  Model saved:       {summary['model_path']}")
    print(f"  Features used:     {summary['num_features']}")
    print(f"  Val retrieval:     {summary['val_retrieval']}")
    print(f"  Val BM25 NDCG@10:  {summary['val_bm25_ranking']['ndcg@10']:.4f}")
    print(f"  Val LGBM NDCG@10:  {summary['val_lgbm_ranking']['ndcg@10']:.4f}")


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
    prepare_parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Raw data directory.")
    prepare_parser.add_argument("--proc-dir", default=str(PROC_DIR), help="Processed data directory.")
    prepare_parser.set_defaults(func=_cmd_prepare)

    search_parser = subparsers.add_parser(
        "search",
        help="Run a product search query.",
    )
    search_parser.add_argument("query", help="Search query string.")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results to display.")
    search_parser.add_argument("--max-length", type=int, default=512, help="Max allowed query length.")
    search_parser.set_defaults(func=_cmd_search)

    # Split parser
    split_parser = subparsers.add_parser(
        "split",
        help="Generate deterministic train/val/test query splits.",
    )
    split_parser.add_argument("--proc-dir", default=str(PROC_DIR), help="Processed data directory.")
    split_parser.add_argument("--val-size", type=float, default=0.15, help="Validation query fraction.")
    split_parser.add_argument("--test-size", type=float, default=0.15, help="Test query fraction.")
    split_parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    split_parser.set_defaults(func=_cmd_split)

    # Training parser
    train_parser = subparsers.add_parser(
        "train",
        help="Run candidate retrieval, feature generation, and LightGBM ranker training.",
    )
    train_parser.add_argument(
        "--proc-dir",
        default=str(PROC_DIR),
        help="Processed data directory.",
    )
    train_parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for trained ranker artifact.",
    )
    train_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Candidate cutoff per query.",
    )
    train_parser.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()