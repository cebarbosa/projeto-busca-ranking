from __future__ import annotations

import argparse
from pathlib import Path

from src.data import materialize_processed
from src.paths import PROC_DIR, RAW_DIR


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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()