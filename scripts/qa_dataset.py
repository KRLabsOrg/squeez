"""Audit and curate a squeez dataset split.

The current JSONL format lets us validate response structure with high confidence:
- response is valid JSON
- response lines appear in the prompt in order
- num_relevant_lines can be corrected exactly

It does not let us recover num_total_lines or compression_ratio exactly for every
sample because the prompt concatenates a potentially multi-paragraph issue text
with raw tool output. Those fields are therefore reported as unreliable but are
left unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _extract_user_body(prompt: str) -> str:
    start = "<|im_start|>user\n"
    end = "<|im_end|>\n<|im_start|>assistant\n"
    if start not in prompt or end not in prompt:
        return ""
    return prompt.split(start, 1)[1].rsplit(end, 1)[0]


def _parse_response(response: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return [], ["invalid_response_json"]

    relevant_lines = payload.get("relevant_lines")
    if not isinstance(relevant_lines, list):
        return [], ["relevant_lines_not_list"]
    if any(not isinstance(line, str) for line in relevant_lines):
        return [], ["relevant_lines_not_all_strings"]
    return relevant_lines, issues


def _check_line_order(haystack: str, lines: list[str]) -> tuple[bool, bool]:
    """Return (all_found, in_order)."""
    cursor = 0
    for line in lines:
        pos = haystack.find(line, cursor)
        if pos == -1:
            if haystack.find(line) == -1:
                return False, False
            return True, False
        cursor = pos + len(line)
    return True, True


def audit_sample(sample: dict) -> tuple[dict, dict]:
    """Audit a single sample and return the curated sample + QA info."""
    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    metadata = dict(sample.get("metadata", {}))
    user_body = _extract_user_body(prompt)

    relevant_lines, issues = _parse_response(response)
    all_found = False
    in_order = False
    if not issues:
        all_found, in_order = _check_line_order(user_body, relevant_lines)
        if not all_found:
            issues.append("relevant_line_missing_from_prompt")
        elif not in_order:
            issues.append("relevant_lines_out_of_order")

    source_num_relevant = metadata.get("num_relevant_lines")
    corrected_num_relevant = len(relevant_lines) if not issues else source_num_relevant
    duplicate_count = len(relevant_lines) - len(set(relevant_lines))

    qa = {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "response_json_valid": "invalid_response_json" not in issues,
        "relevant_lines_present_in_prompt": all_found,
        "relevant_lines_in_prompt_order": in_order,
        "duplicate_relevant_line_count": duplicate_count,
        "source_num_relevant_lines": source_num_relevant,
        "corrected_num_relevant_lines": corrected_num_relevant,
        "source_num_total_lines": metadata.get("num_total_lines"),
        "source_compression_ratio": metadata.get("compression_ratio"),
        "note": (
            "num_total_lines/compression_ratio left unchanged because they cannot "
            "be reconstructed exactly from the assembled prompt format."
        ),
    }

    metadata["num_relevant_lines"] = corrected_num_relevant
    metadata["qa"] = qa

    curated = {
        "prompt": prompt,
        "response": json.dumps({"relevant_lines": relevant_lines}),
        "metadata": metadata,
    }
    return curated, qa


def audit_dataset(input_path: Path) -> tuple[list[dict], dict]:
    """Audit an entire dataset JSONL file."""
    curated_samples: list[dict] = []
    issue_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    corrected_num_relevant = 0
    duplicate_samples = 0

    with open(input_path) as handle:
        samples = [json.loads(line) for line in handle]

    for sample in samples:
        curated, qa = audit_sample(sample)
        curated_samples.append(curated)
        tool_counts[curated["metadata"].get("tool_type", "unknown")] += 1
        issue_counts.update(qa["issues"])
        corrected_num_relevant += (
            qa["source_num_relevant_lines"] != qa["corrected_num_relevant_lines"]
        )
        duplicate_samples += qa["duplicate_relevant_line_count"] > 0

    passing = sum(1 for sample in curated_samples if sample["metadata"]["qa"]["status"] == "pass")
    report = {
        "input_path": str(input_path),
        "total_samples": len(curated_samples),
        "passing_samples": passing,
        "failing_samples": len(curated_samples) - passing,
        "corrected_num_relevant_lines": corrected_num_relevant,
        "samples_with_duplicate_relevant_lines": duplicate_samples,
        "source_num_total_lines_all_zero": all(
            sample["metadata"].get("qa", {}).get("source_num_total_lines") == 0
            for sample in curated_samples
        ),
        "issue_counts": dict(issue_counts),
        "tool_type_counts": dict(tool_counts),
        "notes": [
            "response structure was audited for every sample",
            "num_relevant_lines was corrected exactly from response JSON",
            "num_total_lines and compression_ratio were not rewritten because the assembled prompt does not preserve an exact task/output boundary for every sample",
        ],
    }
    return curated_samples, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and curate a squeez dataset split")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output curated JSONL path")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    curated_samples, report = audit_dataset(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for sample in curated_samples:
            handle.write(json.dumps(sample) + "\n")

    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
