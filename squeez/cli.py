"""Top-level CLI for squeez.

Bare usage defaults to extraction so agent integrations can keep using:

    command | squeez "what matters here?"

Other workflows are available as explicit subcommands.
"""

from __future__ import annotations

import argparse
import sys

from squeez.data import pipeline
from squeez.inference import extractor
from squeez.training import evaluate, train

SUBCOMMANDS = {
    "extract": extractor.main,
    "train": train.main,
    "eval": evaluate.main,
    "pipeline": pipeline.main,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser for squeez."""
    parser = argparse.ArgumentParser(
        prog="squeez",
        description="Filter tool output with a small extractor model, or manage training and data generation.",
        epilog='Bare usage defaults to extraction: `cat output.txt | squeez "find the failing test"`',
    )
    subparsers = parser.add_subparsers(dest="command")

    extractor.build_parser(
        subparsers.add_parser(
            "extract",
            help="Filter tool output with the extractor model",
            description="Extract relevant lines from tool output",
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
    pipeline.build_parser(
        subparsers.add_parser(
            "pipeline",
            help="Run dataset generation and distillation",
        )
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
