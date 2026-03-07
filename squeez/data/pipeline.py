"""CLI orchestrator for the data generation pipeline.

Runs all 8 phases in sequence:
1. Load SWE-bench instances
2. Clone repos and fetch source files
3. Generate tool call specifications
4. Execute real tool calls against cloned repos
5. Auto-label lines using patch ground truth (validation only)
6. LLM distillation (teacher selects relevant spans)
7. Assemble train/eval JSONL
8. Validate dataset quality
"""

import argparse
import logging
import os
import time

from squeez.data.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, phases: list[int] | None = None):
    """Run the data generation pipeline.

    Args:
        config: Pipeline configuration
        phases: List of phase numbers to run (1-8). If None, run all.
    """
    if phases is None:
        phases = list(range(1, 9))

    start_time = time.time()
    logger.info(f"Starting pipeline with phases: {phases}")
    logger.info(f"Output directory: {config.output_dir}")

    instances = None
    all_sources = None
    repo_dirs = None
    tool_calls = None
    tool_outputs = None
    labeled_samples = None
    distilled_samples = None
    train_samples = None
    eval_samples = None

    # Phase 1: Load SWE-bench instances
    if 1 in phases:
        logger.info("=" * 60)
        logger.info("Phase 1: Loading SWE-bench instances")
        logger.info("=" * 60)
        from squeez.data.swebench_loader import load_swebench_instances

        instances = load_swebench_instances(config)
        logger.info(f"Phase 1 complete: {len(instances)} instances")

    # Phase 2: Clone repos and fetch source files
    if 2 in phases:
        logger.info("=" * 60)
        logger.info("Phase 2: Cloning repos and fetching source files")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)

        from squeez.data.source_fetcher import fetch_all_sources

        all_sources, repo_dirs = fetch_all_sources(instances, config)
        logger.info(f"Phase 2 complete: sources for {len(all_sources)} instances")

    # Phase 3: Generate tool call specs
    if 3 in phases:
        logger.info("=" * 60)
        logger.info("Phase 3: Generating tool call specifications")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)
        if all_sources is None:
            from squeez.data.source_fetcher import fetch_all_sources

            all_sources, repo_dirs = fetch_all_sources(instances, config)

        from squeez.data.tool_call_generator import generate_all_tool_calls

        tool_calls = generate_all_tool_calls(instances, all_sources, config)
        logger.info(f"Phase 3 complete: {len(tool_calls)} tool calls")

    # Phase 4: Execute tool calls (real commands on cloned repos)
    if 4 in phases:
        logger.info("=" * 60)
        logger.info("Phase 4: Executing tool calls (real commands)")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)
        if all_sources is None or repo_dirs is None:
            from squeez.data.source_fetcher import fetch_all_sources

            all_sources, repo_dirs = fetch_all_sources(instances, config)
        if tool_calls is None:
            from squeez.data.tool_call_generator import generate_all_tool_calls

            tool_calls = generate_all_tool_calls(instances, all_sources, config)

        from squeez.data.tool_call_executor import execute_all_tool_calls

        tool_outputs = execute_all_tool_calls(tool_calls, repo_dirs, instances, config)
        logger.info(f"Phase 4 complete: {len(tool_outputs)} outputs")

    # Phase 5: Auto-label (optional — used for validation metrics only)
    if 5 in phases:
        logger.info("=" * 60)
        logger.info("Phase 5: Auto-labeling tool outputs (for validation only)")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)
        if tool_outputs is None:
            import json

            tool_outputs_path = config.output_dir / "tool_outputs.jsonl"
            tool_outputs = [json.loads(line) for line in open(tool_outputs_path)]

        from squeez.data.auto_labeler import auto_label_all

        labeled_samples = auto_label_all(tool_outputs, instances, config)
        logger.info(f"Phase 5 complete: {len(labeled_samples)} labeled samples (validation only)")

    # Phase 6: LLM distillation — teacher selects relevant spans
    if 6 in phases:
        logger.info("=" * 60)
        logger.info("Phase 6: LLM distillation (teacher sees task + output only)")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)
        if tool_outputs is None:
            import json

            tool_outputs_path = config.output_dir / "tool_outputs.jsonl"
            tool_outputs = [json.loads(line) for line in open(tool_outputs_path)]

        # Filter to outputs with enough lines to be worth compressing
        from squeez.data.config import MIN_TOTAL_LINES

        distill_inputs = [
            o for o in tool_outputs if len(o["output"].split("\n")) >= MIN_TOTAL_LINES
        ]
        logger.info(f"Distilling {len(distill_inputs)} outputs (>= {MIN_TOTAL_LINES} lines)")

        from squeez.data.llm_distiller import distill_all

        distilled_samples = distill_all(distill_inputs, instances, config)
        logger.info(f"Phase 6 complete: {len(distilled_samples)} distilled samples")

    # Phase 7: Assemble train/eval
    if 7 in phases:
        logger.info("=" * 60)
        logger.info("Phase 7: Assembling train/eval JSONL")
        logger.info("=" * 60)
        if instances is None:
            from squeez.data.swebench_loader import load_swebench_instances

            instances = load_swebench_instances(config)
        if distilled_samples is None:
            import json

            distilled_path = config.output_dir / "distilled_outputs.jsonl"
            distilled_samples = [json.loads(line) for line in open(distilled_path)]

        from squeez.data.sample_assembler import assemble_samples

        train_samples, eval_samples = assemble_samples(distilled_samples, instances, config)
        logger.info(f"Phase 7 complete: {len(train_samples)} train + {len(eval_samples)} eval")

    # Phase 8: Validate
    if 8 in phases:
        logger.info("=" * 60)
        logger.info("Phase 8: Validating dataset")
        logger.info("=" * 60)
        if train_samples is None or eval_samples is None:
            import json

            train_path = config.output_dir / "train.jsonl"
            eval_path = config.output_dir / "eval.jsonl"
            train_samples = [json.loads(line) for line in open(train_path)]
            eval_samples = [json.loads(line) for line in open(eval_path)]

        from squeez.data.validator import validate_dataset, write_validation_report

        report = validate_dataset(train_samples, eval_samples, config)
        report_path = write_validation_report(report, config)
        logger.info(f"Phase 8 complete: report at {report_path}")

        # Print warnings
        for warning in report.get("warnings", []):
            logger.warning(f"  ! {warning}")

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f}s")


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the parser for the data generation pipeline CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Tool Output Extractor — Data Generation Pipeline"
        )

    parser.add_argument(
        "--phase",
        type=int,
        nargs="+",
        default=None,
        help="Phases to run (1-8). Default: all.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="SWE-bench splits to use (default: test)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Quick test mode: limit to N instances",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        default=None,
        help="Directory for cloned repos (default: <output-dir>/repos)",
    )
    parser.add_argument(
        "--teacher-model",
        "--model",
        dest="teacher_model",
        type=str,
        default="gpt-5.4",
        help="Teacher model used for distillation (default: gpt-5.4)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default="",
        help="GitHub API token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--teacher-base-url",
        "--base-url",
        dest="teacher_base_url",
        type=str,
        default=None,
        help="Custom base URL for the teacher model API",
    )
    parser.add_argument(
        "--teacher-api-key",
        "--api-key",
        dest="teacher_api_key",
        type=str,
        default="",
        help="API key for distillation (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls for distillation (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the data generation pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repos_dir = args.repos_dir or f"{args.output_dir}/repos"

    config = PipelineConfig(
        output_dir=args.output_dir,
        source_cache_dir=f"{args.output_dir}/source_cache",
        repos_dir=repos_dir,
        splits=args.splits,
        max_instances=args.test,
        distillation_model=args.teacher_model,
        distillation_base_url=args.teacher_base_url,
        github_token=args.github_token or os.environ.get("GITHUB_TOKEN", ""),
        openai_api_key=args.teacher_api_key or os.environ.get("OPENAI_API_KEY", ""),
        distillation_max_concurrent=args.concurrency,
    )

    run_pipeline(config, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
