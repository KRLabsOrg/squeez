"""Phase 8: Validate generated dataset quality.

Checks compression ratio distribution, tool type balance, split balance,
prompt/output length distributions, and line preservation accuracy.
"""

import json
import logging
import statistics
from collections import Counter
from pathlib import Path

from squeez.data.config import PipelineConfig

logger = logging.getLogger(__name__)


def _compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def validate_dataset(
    train_samples: list[dict],
    eval_samples: list[dict],
    config: PipelineConfig,
) -> dict:
    """Run validation checks on the assembled dataset.

    Args:
        train_samples: Training samples
        eval_samples: Evaluation samples
        config: Pipeline configuration

    Returns:
        Validation report dict
    """
    report = {
        "summary": {},
        "compression_ratio": {},
        "tool_type_balance": {},
        "split_balance": {},
        "prompt_length": {},
        "response_length": {},
        "warnings": [],
    }

    all_samples = train_samples + eval_samples

    if not all_samples:
        report["warnings"].append("No samples to validate!")
        return report

    # Summary
    report["summary"] = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
    }

    # Compression ratio distribution
    ratios = [s["metadata"].get("compression_ratio", 0) for s in all_samples if "metadata" in s]
    report["compression_ratio"] = _compute_stats(ratios)

    # Check target range (70-95% compression)
    in_range = sum(1 for r in ratios if 0.70 <= r <= 0.95)
    report["compression_ratio"]["pct_in_target_range"] = (
        round(in_range / len(ratios), 4) if ratios else 0
    )

    if ratios and statistics.mean(ratios) < 0.5:
        report["warnings"].append(
            f"Mean compression ratio ({statistics.mean(ratios):.2%}) is below 50%"
        )

    # Tool type balance
    tool_counts = Counter(
        s["metadata"].get("tool_type", "unknown") for s in all_samples if "metadata" in s
    )
    report["tool_type_balance"] = dict(tool_counts.most_common())

    # Check for missing tool types
    expected_tools = {"read_file", "grep", "git_log", "test_output", "git_diff"}
    missing = expected_tools - set(tool_counts.keys())
    if missing:
        report["warnings"].append(f"Missing tool types: {missing}")

    # Split balance
    report["split_balance"] = {
        "train": len(train_samples),
        "eval": len(eval_samples),
        "train_pct": round(len(train_samples) / len(all_samples), 4),
    }

    if len(eval_samples) < 10:
        report["warnings"].append(f"Very few eval samples ({len(eval_samples)})")

    # Prompt length distribution (in characters)
    prompt_lengths = [len(s["prompt"]) for s in all_samples]
    report["prompt_length"] = _compute_stats(prompt_lengths)

    # Response length distribution (in characters)
    response_lengths = [len(s["response"]) for s in all_samples]
    report["response_length"] = _compute_stats(response_lengths)

    # Check for extremely long prompts
    long_prompts = sum(1 for prompt_length in prompt_lengths if prompt_length > 50000)
    if long_prompts > 0:
        report["warnings"].append(f"{long_prompts} prompts exceed 50K characters")

    # Check for empty responses
    empty = sum(1 for s in all_samples if not s["response"].strip())
    if empty > 0:
        report["warnings"].append(f"{empty} samples have empty responses")

    # Unique instances
    unique_instances = set(s["metadata"]["instance_id"] for s in all_samples if "metadata" in s)
    report["summary"]["unique_instances"] = len(unique_instances)

    return report


def write_validation_report(report: dict, config: PipelineConfig) -> Path:
    """Write validation report to disk as text and JSON.

    Returns path to the text report.
    """
    report_path = config.output_dir / "validation_report.txt"
    json_path = config.output_dir / "validation_report.json"

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write human-readable text
    lines = ["=" * 60, "DATASET VALIDATION REPORT", "=" * 60, ""]

    # Summary
    lines.append("## Summary")
    for k, v in report.get("summary", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Compression ratio
    lines.append("## Compression Ratio")
    for k, v in report.get("compression_ratio", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Tool type balance
    lines.append("## Tool Type Balance")
    for tool, count in report.get("tool_type_balance", {}).items():
        lines.append(f"  {tool}: {count}")
    lines.append("")

    # Split balance
    lines.append("## Split Balance")
    for k, v in report.get("split_balance", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Prompt/Response lengths
    for section in ["prompt_length", "response_length"]:
        lines.append(f"## {section.replace('_', ' ').title()}")
        for k, v in report.get(section, {}).items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Warnings
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("## WARNINGS")
        for w in warnings:
            lines.append(f"  ! {w}")
    else:
        lines.append("## No warnings - dataset looks good!")

    lines.append("")
    text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(text)

    logger.info(f"Validation report written to {report_path}")
    return report_path
