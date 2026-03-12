"""Prepare encoder training data from assembled splits or the HF dataset.

Converts the generative-format train/dev/test JSONL (prompt + response) into
encoder-format JSONL (task + tool_output + relevant_lines + tool_type).

Uses either the released HuggingFace dataset or local assembled train/dev/test
files, so the encoder trains and evaluates on the same splits as the
generative model.

Usage:
    python scripts/prepare_encoder_data.py --data-dir data
    python scripts/prepare_encoder_data.py --from-hf
    python scripts/prepare_encoder_data.py --data-dir data --output-dir data/encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_from_prompt(prompt: str) -> tuple[str, str]:
    """Extract query/task and tool_output from a ChatML-formatted prompt."""
    task_m = re.search(r"<query>\s*\n?(.*?)\n?\s*</query>", prompt, re.DOTALL)
    if not task_m:
        task_m = re.search(r"<task>\s*\n?(.*?)\n?\s*</task>", prompt, re.DOTALL)
    out_m = re.search(r"<tool_output>\s*\n?(.*?)\n?\s*</tool_output>", prompt, re.DOTALL)
    task = task_m.group(1).strip() if task_m else ""
    tool_output = out_m.group(1).strip() if out_m else ""
    return task, tool_output


def _extract_relevant_lines(response: str) -> list[str]:
    """Extract relevant lines from an XML-formatted response."""
    m = re.search(r"<relevant_lines>\s*\n?(.*?)\n?\s*</relevant_lines>", response, re.DOTALL)
    if not m:
        return []
    content = m.group(1).strip()
    if not content:
        return []
    lines = []
    omit_pattern = re.compile(r"^\.\.\.\s*\(\d+\s+lines?\s+omitted\)\s*$")
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if omit_pattern.match(stripped):
            continue
        lines.append(line)
    return lines


def convert_split(samples: list[dict]) -> list[dict]:
    """Convert generative-format samples to encoder format."""
    converted = []
    skipped = 0

    for sample in samples:
        if {"query", "tool_output", "gold_spans", "tool_type"} <= set(sample):
            from squeez.data.canonical import extract_relevant_lines

            converted.append(
                {
                    "task": sample["query"],
                    "tool_output": sample["tool_output"],
                    "relevant_lines": extract_relevant_lines(
                        sample["tool_output"], sample["gold_spans"]
                    ),
                    "tool_type": sample["tool_type"],
                }
            )
            continue

        task, tool_output = _extract_from_prompt(sample["prompt"])

        if not task or not tool_output:
            skipped += 1
            continue

        relevant_lines = _extract_relevant_lines(sample["response"])

        metadata = sample.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        converted.append(
            {
                "task": task,
                "tool_output": tool_output,
                "relevant_lines": relevant_lines,
                "tool_type": metadata.get("tool_type", "unknown"),
            }
        )

    if skipped:
        logger.warning(f"Skipped {skipped} samples with missing task/output")
    return converted


def write_split(samples: list[dict], output_path: Path) -> int:
    """Write encoder samples to JSONL."""
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare encoder training data from assembled splits"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.jsonl/dev.jsonl/test.jsonl (default: download from HF)",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Download splits from HuggingFace instead of using local files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for encoder_*.jsonl output (default: data)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.from_hf or args.data_dir is None:
        logger.info("Downloading splits from HuggingFace...")
        from datasets import load_dataset

        ds = load_dataset("KRLabsOrg/tool-output-extraction-swebench")
        splits = {
            "train": [dict(row) for row in ds["train"]],
            "dev": [dict(row) for row in ds["dev"]],
            "test": [dict(row) for row in ds["test"]],
        }
    else:
        splits = {}
        for name in ["train", "dev", "test"]:
            path = args.data_dir / f"{name}.jsonl"
            if not path.exists():
                parser.error(f"Missing {path}")
            with open(path) as f:
                splits[name] = [json.loads(line) for line in f if line.strip()]
        logger.info(
            f"Loaded from {args.data_dir}: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, samples in splits.items():
        converted = convert_split(samples)

        # Stats
        tool_stats = defaultdict(lambda: {"total": 0, "empty": 0})
        for s in converted:
            tt = s["tool_type"]
            tool_stats[tt]["total"] += 1
            if not s["relevant_lines"]:
                tool_stats[tt]["empty"] += 1

        total_empty = sum(d["empty"] for d in tool_stats.values())
        pct = 100 * total_empty / len(converted) if converted else 0

        path = args.output_dir / f"encoder_{name}.jsonl"
        n = write_split(converted, path)
        logger.info(f"{name}: {n} samples ({total_empty} empty, {pct:.1f}%) -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
