"""Filter distilled_outputs.jsonl to cap empty samples at a target ratio per tool type.

Keeps all non-empty samples. Randomly downsamples empties so each tool type
has at most --max-empty-ratio empty samples (default 10%).

Usage:
    python scripts/filter_distilled.py data/v2/distilled_outputs.jsonl
    python scripts/filter_distilled.py data/v2/distilled_outputs.jsonl --max-empty-ratio 0.10
    python scripts/filter_distilled.py data/v2/distilled_outputs.jsonl --output data/v2/distilled_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def filter_distilled(
    input_path: Path,
    output_path: Path | None = None,
    max_empty_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    if output_path is None:
        output_path = input_path

    # Load and split by tool type
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: {"non_empty": [], "empty": []})

    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            tt = sample["tool_type"]
            has_content = bool(sample.get("spans"))
            bucket = "non_empty" if has_content else "empty"
            by_type[tt][bucket].append(sample)

    # Filter
    rng = random.Random(seed)
    kept = []
    total_before = 0
    total_after = 0
    total_dropped = 0

    logger.info(
        f"{'tool_type':<20s} {'non_empty':>10s} {'empty':>8s} {'kept_empty':>12s} {'dropped':>9s}"
    )

    for tt in sorted(by_type):
        non_empty = by_type[tt]["non_empty"]
        empty = by_type[tt]["empty"]
        total_before += len(non_empty) + len(empty)

        # Cap empties: allow at most max_empty_ratio of total for this tool
        # target_empty / (non_empty + target_empty) <= max_empty_ratio
        # target_empty <= max_empty_ratio * non_empty / (1 - max_empty_ratio)
        if non_empty:
            max_empty = int(max_empty_ratio * len(non_empty) / (1 - max_empty_ratio))
        else:
            max_empty = 0

        if len(empty) > max_empty:
            rng.shuffle(empty)
            kept_empty = empty[:max_empty]
            dropped = len(empty) - max_empty
        else:
            kept_empty = empty
            dropped = 0

        kept.extend(non_empty)
        kept.extend(kept_empty)
        total_after += len(non_empty) + len(kept_empty)
        total_dropped += dropped

        logger.info(
            f"  {tt:<20s} {len(non_empty):>8d} {len(empty):>8d} {len(kept_empty):>10d} {dropped:>9d}"
        )

    logger.info(f"\nTotal: {total_before} -> {total_after} ({total_dropped} empties dropped)")

    # Write
    with open(output_path, "w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")

    logger.info(f"Written to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter distilled outputs to cap empty ratio")
    parser.add_argument("input", type=Path, help="Path to distilled_outputs.jsonl")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path (default: overwrite input)"
    )
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=0.10,
        help="Max empty ratio per tool (default 0.10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    filter_distilled(args.input, args.output, args.max_empty_ratio, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
