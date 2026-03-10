"""QA script for synthetic encoder-format data.

Validates synthetic JSONL samples: checks relevant_lines exist in tool_output,
no duplicates, correct order, and optionally runs LLM-assisted review.

Adapted from scripts/qa_dataset.py for encoder-format data
({task, tool_output, relevant_lines, tool_type}).

Usage:
    # Automated checks only
    python scripts/qa_synthetic_data.py --input data/synthetic_train.jsonl --report data/qa_report.json

    # With LLM-assisted review
    python scripts/qa_synthetic_data.py --input data/synthetic_train.jsonl --report data/qa_report.json --llm-review

    # Flag samples for manual correction
    python scripts/qa_synthetic_data.py --input data/synthetic_train.jsonl --flagged data/flagged.jsonl --report data/qa_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    """Strip and collapse whitespace for fuzzy matching."""
    return " ".join(s.split())


def _check_lines_in_output(tool_output: str, relevant_lines: list[str]) -> tuple[bool, list[str]]:
    """Check that all relevant lines appear in the tool output.

    Returns (all_found, missing_lines).
    """
    missing = []
    for line in relevant_lines:
        if not line.strip():
            continue
        if line in tool_output:
            continue
        # Try normalized match
        norm_line = _normalize(line)
        norm_output = _normalize(tool_output)
        if norm_line not in norm_output:
            missing.append(line)

    return len(missing) == 0, missing


def _check_line_order(tool_output: str, relevant_lines: list[str]) -> bool:
    """Check that relevant lines appear in order within the tool output."""
    cursor = 0
    for line in relevant_lines:
        if not line.strip():
            continue
        pos = tool_output.find(line, cursor)
        if pos == -1:
            # Try from start (out of order)
            if tool_output.find(line) != -1:
                return False
            continue  # Missing lines handled separately
        cursor = pos + len(line)
    return True


def _check_duplicates(relevant_lines: list[str]) -> list[str]:
    """Return list of duplicate relevant lines."""
    seen = set()
    dupes = []
    for line in relevant_lines:
        if line in seen:
            dupes.append(line)
        seen.add(line)
    return dupes


def audit_sample(sample: dict, idx: int) -> dict:
    """Audit a single encoder-format sample.

    Returns a QA info dict with issues found.
    """
    task = sample.get("task", "")
    tool_output = sample.get("tool_output", "")
    relevant_lines = sample.get("relevant_lines", [])
    tool_type = sample.get("tool_type", "unknown")

    issues = []

    # Basic structure checks
    if not task or len(task) < 10:
        issues.append("task_too_short")
    if not tool_output or len(tool_output) < 20:
        issues.append("tool_output_too_short")
    if not isinstance(relevant_lines, list):
        issues.append("relevant_lines_not_list")
        return {
            "idx": idx,
            "tool_type": tool_type,
            "status": "fail",
            "issues": issues,
        }

    # Check for non-string entries
    non_strings = [i for i, ln in enumerate(relevant_lines) if not isinstance(ln, str)]
    if non_strings:
        issues.append(f"relevant_lines_not_all_strings (indices: {non_strings})")

    # Check lines appear in output
    all_found, missing = _check_lines_in_output(tool_output, relevant_lines)
    if not all_found:
        issues.append(f"relevant_lines_missing ({len(missing)} missing)")

    # Check order
    if all_found and not _check_line_order(tool_output, relevant_lines):
        issues.append("relevant_lines_out_of_order")

    # Check duplicates
    dupes = _check_duplicates(relevant_lines)
    if dupes:
        issues.append(f"duplicate_relevant_lines ({len(dupes)} dupes)")

    # Check relevance ratio
    output_lines = [ln for ln in tool_output.split("\n") if ln.strip()]
    n_output = len(output_lines)
    n_relevant = len(relevant_lines)
    if n_output > 0 and n_relevant > 0:
        ratio = n_relevant / n_output
        if ratio > 0.5:
            issues.append(f"high_relevant_ratio ({ratio:.0%})")

    return {
        "idx": idx,
        "tool_type": tool_type,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "n_output_lines": n_output,
        "n_relevant_lines": n_relevant,
        "missing_lines": missing if not all_found else [],
        "duplicate_count": len(dupes),
    }


def _llm_review_sample(client, sample: dict, model: str) -> dict | None:
    """Run LLM-assisted review on a single sample."""
    task = sample["task"]
    tool_output = sample["tool_output"]
    relevant_lines = sample["relevant_lines"]

    # Truncate if too long
    truncated = tool_output[:3000] + "\n..." if len(tool_output) > 3000 else tool_output
    lines_json = json.dumps(relevant_lines, indent=2)

    prompt = f"""Review the label quality for this tool output extraction sample.

Task: {task}

Tool output:
{truncated}

Labeled relevant lines:
{lines_json}

Assess:
1. Are all labeled lines actually relevant to the task?
2. Are there important lines that were missed?
3. Overall label quality (0.0 = terrible, 1.0 = perfect)

Return JSON: {{"quality_score": 0.0-1.0, "false_positives": ["line..."], "missed_lines": ["line..."], "notes": "..."}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        text = response.choices[0].message.content

        # Parse JSON response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.debug(f"LLM review failed: {e}")

    return None


def audit_dataset(
    input_path: Path,
    llm_review: bool = False,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Audit an entire synthetic dataset.

    Returns (samples, qa_results, summary_report).
    """
    with open(input_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    qa_results = []
    issue_counts: Counter = Counter()
    type_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})

    for i, sample in enumerate(samples):
        qa = audit_sample(sample, i)
        qa_results.append(qa)
        issue_counts.update(qa["issues"])
        type_stats[qa["tool_type"]]["total"] += 1
        type_stats[qa["tool_type"]][qa["status"]] += 1

    # LLM-assisted review on flagged samples
    llm_report = {}
    if llm_review:
        from openai import OpenAI

        client = OpenAI(base_url=base_url) if base_url else OpenAI()

        flagged_indices = [qa["idx"] for qa in qa_results if qa["status"] == "fail"]
        # Also review a random 10% of passing samples
        import random

        rng = random.Random(42)
        passing = [qa["idx"] for qa in qa_results if qa["status"] == "pass"]
        review_passing = rng.sample(passing, min(len(passing) // 10, 50))

        to_review = flagged_indices + review_passing
        logger.info(f"Running LLM review on {len(to_review)} samples...")

        reviewed = 0
        for idx in to_review:
            result = _llm_review_sample(client, samples[idx], model)
            if result:
                qa_results[idx]["llm_review"] = result
                reviewed += 1
            if reviewed % 10 == 0:
                logger.info(f"  Reviewed {reviewed}/{len(to_review)}")

        llm_report = {
            "reviewed": reviewed,
            "avg_quality": (
                sum(
                    qa.get("llm_review", {}).get("quality_score", 0)
                    for qa in qa_results
                    if "llm_review" in qa
                )
                / max(reviewed, 1)
            ),
        }

    passing = sum(1 for qa in qa_results if qa["status"] == "pass")
    report = {
        "input_path": str(input_path),
        "total_samples": len(samples),
        "passing": passing,
        "failing": len(samples) - passing,
        "pass_rate": passing / len(samples) if samples else 0,
        "issue_counts": dict(issue_counts),
        "per_type_stats": dict(type_stats),
    }
    if llm_report:
        report["llm_review"] = llm_report

    return samples, qa_results, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA synthetic encoder-format data")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--flagged", default=None, help="Output JSONL of flagged samples")
    parser.add_argument("--llm-review", action="store_true", help="Run LLM-assisted review")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for review")
    parser.add_argument("--base-url", default=None, help="Custom API base URL")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_path = Path(args.input)
    report_path = Path(args.report)

    samples, qa_results, report = audit_dataset(
        input_path,
        llm_review=args.llm_review,
        model=args.model,
        base_url=args.base_url,
    )

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write flagged samples if requested
    if args.flagged:
        flagged_path = Path(args.flagged)
        flagged_path.parent.mkdir(parents=True, exist_ok=True)
        flagged = [{**samples[qa["idx"]], "qa": qa} for qa in qa_results if qa["status"] == "fail"]
        with open(flagged_path, "w") as f:
            for s in flagged:
                f.write(json.dumps(s) + "\n")
        logger.info(f"Flagged {len(flagged)} samples -> {flagged_path}")

    # Print summary
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
