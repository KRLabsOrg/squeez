"""Apply high-confidence traceback fixes to the training split.

This pass is intentionally conservative:
- only `python` samples with empty or very short selected lines are touched
- only `test_output` samples with very short selected lines are touched
- only when the raw output clearly contains a traceback or import/error block
"""

import argparse
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^[=\-_*#]{6,}$")
TERMINAL_RE = re.compile(r"(Error|Exception|Warning):")


def _extract_raw_output(prompt: str) -> str:
    body = prompt.split("<|im_start|>user\n", 1)[1].rsplit(
        "<|im_end|>\n<|im_start|>assistant\n", 1
    )[0]
    return body.split("\n\n", 1)[1] if "\n\n" in body else body


def _looks_like_python_candidate(
    tool_type: str, relevant_lines: list[str], raw_output: str
) -> bool:
    if tool_type != "python":
        return False
    if not relevant_lines:
        return (
            "Traceback (most recent call last):" in raw_output
            or "ERROR: while parsing the following warning configuration:" in raw_output
            or "INTERNALERROR>" in raw_output
            or "ImportError while loading conftest" in raw_output
        )
    return len(relevant_lines) <= 5 and "Traceback (most recent call last):" in raw_output


def _looks_like_test_output_candidate(
    tool_type: str, relevant_lines: list[str], raw_output: str
) -> bool:
    if tool_type != "test_output" or not relevant_lines or len(relevant_lines) > 4:
        return False
    return (
        "ImportError while loading conftest" in raw_output
        or "==================================== ERRORS ===================================="
        in raw_output
    )


def _find_python_start(lines: list[str]) -> int | None:
    priority_prefixes = [
        "ERROR: while parsing the following warning configuration:",
        "ImportError while loading conftest",
    ]
    for prefix in priority_prefixes:
        matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if matches:
            return matches[-1]

    traceback_matches = [
        i for i, line in enumerate(lines) if "Traceback (most recent call last):" in line
    ]
    if traceback_matches:
        return traceback_matches[-1]

    internal_matches = [i for i, line in enumerate(lines) if line.startswith("INTERNALERROR>")]
    if internal_matches:
        return internal_matches[0]

    return None


def _find_terminal_line(lines: list[str], start_idx: int) -> int:
    last_nonempty = start_idx
    last_terminal = start_idx

    for i in range(start_idx, len(lines)):
        line = lines[i]
        if line.strip():
            last_nonempty = i
        stripped = line.strip()
        if (
            stripped.startswith("E   ")
            or stripped.startswith("ERROR:")
            or stripped.startswith("INTERNALERROR>")
            or TERMINAL_RE.search(stripped)
        ):
            last_terminal = i

    end_idx = last_nonempty
    for i in range(last_terminal + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if HEADING_RE.match(line.strip()):
            end_idx = i - 1
            while end_idx > start_idx and not lines[end_idx].strip():
                end_idx -= 1
            return end_idx

    return end_idx


def _extract_python_lines(raw_output: str) -> list[str]:
    lines = raw_output.splitlines()
    start_idx = _find_python_start(lines)
    if start_idx is None:
        return []
    end_idx = _find_terminal_line(lines, start_idx)
    return lines[start_idx : end_idx + 1]


def _extract_test_output_lines(raw_output: str) -> list[str]:
    lines = raw_output.splitlines()
    error_header_matches = [
        i
        for i, line in enumerate(lines)
        if line.startswith(
            "==================================== ERRORS ===================================="
        )
    ]
    if error_header_matches:
        start_idx = error_header_matches[-1]
    else:
        import_matches = [
            i
            for i, line in enumerate(lines)
            if line.startswith("ImportError while loading conftest")
        ]
        if import_matches:
            start_idx = import_matches[-1]
        else:
            start_idx = 0

    summary_matches = [
        i
        for i, line in enumerate(lines[start_idx:], start=start_idx)
        if re.match(r"^\d+ error[s]? in ", line.strip())
    ]
    if summary_matches:
        end_idx = summary_matches[-1]
    else:
        end_idx = _find_terminal_line(lines, start_idx)

    return lines[start_idx : end_idx + 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply conservative traceback fixes to the training split"
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--notes", required=True, help="Output notes JSON path")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.input).open()]
    corrected_indices: list[int] = []
    corrected_by_tool = {"python": 0, "test_output": 0}
    notes: list[dict] = []

    for index, sample in enumerate(rows):
        raw_output = _extract_raw_output(sample["prompt"])
        tool_type = sample["metadata"]["tool_type"]
        relevant_lines = json.loads(sample["response"])["relevant_lines"]
        status = "reviewed_no_change"

        if _looks_like_python_candidate(tool_type, relevant_lines, raw_output):
            new_lines = _extract_python_lines(raw_output)
            if new_lines and new_lines != relevant_lines:
                sample["response"] = json.dumps({"relevant_lines": new_lines})
                sample["metadata"]["num_relevant_lines"] = len(new_lines)
                corrected_indices.append(index)
                corrected_by_tool["python"] += 1
                status = "corrected_traceback_candidate"

        elif _looks_like_test_output_candidate(tool_type, relevant_lines, raw_output):
            new_lines = _extract_test_output_lines(raw_output)
            if new_lines and new_lines != relevant_lines:
                sample["response"] = json.dumps({"relevant_lines": new_lines})
                sample["metadata"]["num_relevant_lines"] = len(new_lines)
                corrected_indices.append(index)
                corrected_by_tool["test_output"] += 1
                status = "corrected_traceback_candidate"

        sample.setdefault("metadata", {}).setdefault("qa", {})["traceback_train_curation_v1"] = {
            "status": status,
            "rationale": (
                "Conservative traceback-based curation pass. "
                "Only empty or clearly truncated python/test_output labels were expanded."
            ),
        }
        if status != "reviewed_no_change":
            notes.append(
                {
                    "index": index,
                    "instance_id": sample["metadata"]["instance_id"],
                    "tool_type": tool_type,
                    "num_relevant_lines": sample["metadata"]["num_relevant_lines"],
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    payload = {
        "input_path": args.input,
        "output_path": args.output,
        "summary": {
            "total_samples": len(rows),
            "corrected_total": len(corrected_indices),
            "corrected_python": corrected_by_tool["python"],
            "corrected_test_output": corrected_by_tool["test_output"],
        },
        "corrected_indices": corrected_indices,
        "notes": notes,
    }
    with open(args.notes, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
