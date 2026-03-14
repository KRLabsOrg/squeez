"""Top-level CLI for squeez.

Bare usage defaults to extraction so agent integrations can keep using:

    command | squeez "what matters here?"

Other workflows are available as explicit subcommands.
"""

from __future__ import annotations

import argparse
import sys

from squeez.inference import extractor
from squeez.training import evaluate, train


def _build_dataset_main(argv: list[str] | None = None) -> int:
    from scripts.build_full_dataset import main as build_dataset_main

    return build_dataset_main(argv)


SUBCOMMANDS = {
    "extract": extractor.main,
    "train": train.main,
    "eval": evaluate.main,
    "build-dataset": _build_dataset_main,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser for squeez."""
    parser = argparse.ArgumentParser(
        prog="squeez",
        description="Filter tool output with a small extractor model, or manage training and dataset generation.",
        epilog='Bare usage defaults to extraction: `cat output.txt | squeez "find the failing test"`',
    )
    subparsers = parser.add_subparsers(dest="command")

    extractor.build_parser(
        subparsers.add_parser(
            "extract",
            help="Filter tool output with the extractor model",
            description="Extract the relevant evidence block from tool output",
        )
    )
    train.build_parser(
        subparsers.add_parser(
            "train",
            help="Fine-tune an extractor model with LoRA",
        )
    )
    evaluate.build_parser(
        subparsers.add_parser(
            "eval",
            help="Evaluate a trained extractor model",
        )
    )
    build_parser = argparse.ArgumentParser(
        add_help=False,
    )
    build_parser.add_argument("--output-dir", type=str, default="data")
    build_parser.add_argument("--repos-dir", type=str, default=None)
    build_parser.add_argument("--splits", nargs="+", default=["test"])
    build_parser.add_argument("--test", type=int, default=None)
    build_parser.add_argument("--teacher-model", default="openai/gpt-oss-120b")
    build_parser.add_argument("--teacher-base-url", default="http://localhost:8000/v1")
    build_parser.add_argument("--teacher-api-key", default="")
    build_parser.add_argument("--github-token", default="")
    build_parser.add_argument("--concurrency", type=int, default=10)
    build_parser.add_argument("--synthetic-config", default="configs/synthetic_tools.yaml")
    build_parser.add_argument("--synthetic-small-batch", action="store_true")
    build_parser.add_argument("--synthetic-validate", action="store_true")
    build_parser.add_argument("--synthetic-temperature", type=float, default=0.7)
    build_parser.add_argument("--synthetic-concurrency", type=int, default=10)
    build_parser.add_argument("--synthetic-tool-types", nargs="+", default=None)
    subparsers.add_parser(
        "build-dataset",
        help="Build the dataset from scratch (SWE + synthetic)",
        parents=[build_parser],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for squeez."""
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in SUBCOMMANDS:
        command = argv[0]
        return SUBCOMMANDS[command](argv[1:])

    if not argv or argv[0] in {"-h", "--help"}:
        parser = build_parser()
        parser.print_help()
        return 0

    return extractor.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
