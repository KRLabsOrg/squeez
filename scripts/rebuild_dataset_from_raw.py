"""Relabel reused raw SWE + synthetic outputs and assemble final dataset splits.

This is the main end-to-end regeneration entrypoint for the v3 dataset shape.
It reuses raw tool outputs, regenerates focused queries and canonical spans,
optionally adds synthetic negatives, then assembles canonical/Qwen/encoder
train-dev-test files in one output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from squeez.data.config import PipelineConfig
from squeez.data.sample_assembler import assemble_samples

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from relabel_raw_outputs import relabel_async  # noqa: E402

logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def rebuild_async(
    *,
    swe_input: Path,
    swe_task_source: Path | None,
    synthetic_input: Path,
    output_dir: Path,
    model: str,
    base_url: str | None,
    concurrency: int,
    negative_ratio: float,
    seed: int,
) -> int:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    swe_output = raw_dir / "canonical_swe.jsonl"
    synth_output = raw_dir / "canonical_synth.jsonl"

    await relabel_async(
        input_path=swe_input,
        output_path=swe_output,
        model=model,
        base_url=base_url,
        task_source=swe_task_source,
        default_source="swe",
        concurrency=concurrency,
        add_negatives=False,
        negative_ratio=0.0,
        seed=seed,
    )
    await relabel_async(
        input_path=synthetic_input,
        output_path=synth_output,
        model=model,
        base_url=base_url,
        task_source=None,
        default_source="synthetic",
        concurrency=concurrency,
        add_negatives=True,
        negative_ratio=negative_ratio,
        seed=seed,
    )

    swe_rows = _load_rows(swe_output)
    synth_rows = _load_rows(synth_output)
    combined = swe_rows + synth_rows

    combined_path = output_dir / "distilled_outputs.jsonl"
    with open(combined_path, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d combined canonical rows to %s", len(combined), combined_path)

    config = PipelineConfig(output_dir=output_dir)
    assemble_samples(combined, [], config, force_rebuild=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild dataset splits from raw SWE + synthetic tool outputs"
    )
    parser.add_argument("--swe-input", type=Path, required=True)
    parser.add_argument("--swe-task-source", type=Path, default=None)
    parser.add_argument("--synthetic-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(
        rebuild_async(
            swe_input=args.swe_input,
            swe_task_source=args.swe_task_source,
            synthetic_input=args.synthetic_input,
            output_dir=args.output_dir,
            model=args.model,
            base_url=args.base_url,
            concurrency=args.concurrency,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
