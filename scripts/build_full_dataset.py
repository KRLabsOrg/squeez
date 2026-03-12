"""Build the full dataset from scratch: SWE pipeline + synthetic generation.

This is the clean public entrypoint for fresh generation. It does not depend on
existing v2 artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from scripts.generate_synthetic_data import generate_all_async
from squeez.data.config import PipelineConfig
from squeez.data.pipeline import run_pipeline
from squeez.data.sample_assembler import assemble_samples

logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def build_async(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repos_dir = Path(args.repos_dir) if args.repos_dir else output_dir / "repos"

    config = PipelineConfig(
        output_dir=output_dir,
        source_cache_dir=output_dir / "source_cache",
        repos_dir=repos_dir,
        splits=args.splits,
        max_instances=args.test,
        distillation_model=args.teacher_model,
        distillation_base_url=args.teacher_base_url,
        github_token=args.github_token or os.environ.get("GITHUB_TOKEN", ""),
        openai_api_key=args.teacher_api_key or os.environ.get("OPENAI_API_KEY", ""),
        distillation_max_concurrent=args.concurrency,
    )

    logger.info("Running SWE pipeline phases 1-6 into %s", output_dir)
    run_pipeline(config, phases=[1, 2, 3, 4, 5, 6])

    synthetic_path = output_dir / "synthetic_train.jsonl"
    logger.info("Generating synthetic canonical rows into %s", synthetic_path)
    await generate_all_async(
        config_path=Path(args.synthetic_config),
        output_path=synthetic_path,
        model=args.teacher_model,
        base_url=args.teacher_base_url,
        small_batch=args.synthetic_small_batch,
        validate=args.synthetic_validate,
        temperature=args.synthetic_temperature,
        concurrency=args.synthetic_concurrency,
        tool_types=args.synthetic_tool_types,
    )

    swe_distilled = _load_rows(output_dir / "distilled_outputs.jsonl")
    synthetic_rows = _load_rows(synthetic_path)
    combined = swe_distilled + synthetic_rows

    combined_path = output_dir / "distilled_outputs_combined.jsonl"
    with open(combined_path, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d combined canonical rows to %s", len(combined), combined_path)

    assemble_samples(combined, [], config, force_rebuild=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build full dataset from scratch (SWE + synthetic)"
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--repos-dir", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--teacher-model", default="openai/gpt-oss-120b")
    parser.add_argument("--teacher-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--github-token", default="")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--synthetic-config",
        default="configs/synthetic_tools.yaml",
    )
    parser.add_argument("--synthetic-small-batch", action="store_true")
    parser.add_argument("--synthetic-validate", action="store_true")
    parser.add_argument("--synthetic-temperature", type=float, default=0.7)
    parser.add_argument("--synthetic-concurrency", type=int, default=10)
    parser.add_argument("--synthetic-tool-types", nargs="+", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(build_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
