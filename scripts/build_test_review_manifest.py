from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def parse_prompt_response(sample: dict) -> tuple[str, str, list[str]]:
    prompt = sample["prompt"]
    response = sample["response"]
    task = prompt.split("<task>\n", 1)[1].split("\n</task>\n<tool_output>\n", 1)[0]
    tool_output = prompt.split("\n</task>\n<tool_output>\n", 1)[1].split(
        "\n</tool_output><|im_end|>", 1
    )[0]
    relevant_block = response.split("<relevant_lines>\n", 1)[1].split("\n</relevant_lines>", 1)[0]
    relevant_lines = [line for line in relevant_block.split("\n") if line.strip()]
    return task, tool_output, relevant_lines


def raw_relevant_lines(sample: dict) -> list[str]:
    if not sample.get("spans"):
        return []
    return [
        line
        for line in sample["distilled_output"].split("\n")
        if line.strip() and not line.strip().startswith("...")
    ]


def build_raw_index(raw_paths: list[Path]) -> dict[tuple[str, str, str, str], dict]:
    index: dict[tuple[str, str, str, str], dict] = {}
    for raw_path in raw_paths:
        with raw_path.open() as handle:
            for row_idx, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                key = (
                    sample["instance_id"],
                    sample["tool_type"],
                    sha1(sample["output"]),
                    sha1("\n".join(raw_relevant_lines(sample))),
                )
                index[key] = {
                    "raw_file": str(raw_path),
                    "raw_row": row_idx,
                    "num_lines": sample.get("num_lines"),
                    "distilled_lines": sample.get("distilled_lines"),
                    "kept_lines": sample.get("kept_lines"),
                    "num_spans": len(sample.get("spans", [])),
                }
    return index


def classify(sample: dict, relevant_lines: list[str]) -> list[str]:
    flags: list[str] = []
    meta = sample["metadata"]
    tool_type = meta["tool_type"]
    total = meta["num_total_lines"]
    rel = meta["num_relevant_lines"]
    if rel == 0:
        flags.append("empty")
    if rel >= 100:
        flags.append("very_large_positive")
    if total >= 500:
        flags.append("max_window_source")
    if tool_type in {"grep", "ls", "type_check", "git_blame", "read_file"}:
        flags.append(f"tool:{tool_type}")
    if rel and total and rel / total > 0.5:
        flags.append("high_coverage")
    if len(relevant_lines) != rel:
        flags.append("metadata_mismatch")
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--raw-file",
        type=Path,
        action="append",
        required=True,
        help="Raw source file(s) used to assemble the test split",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw_index = build_raw_index(args.raw_file)
    manifest = []
    with args.test_file.open() as handle:
        for row_idx, line in enumerate(handle, 1):
            sample = json.loads(line)
            task, tool_output, relevant_lines = parse_prompt_response(sample)
            meta = sample["metadata"]
            key = (
                meta["instance_id"],
                meta["tool_type"],
                sha1(tool_output),
                sha1("\n".join(relevant_lines)),
            )
            manifest.append(
                {
                    "test_row": row_idx,
                    "instance_id": meta["instance_id"],
                    "tool_type": meta["tool_type"],
                    "source": meta["source"],
                    "num_total_lines": meta["num_total_lines"],
                    "num_relevant_lines": meta["num_relevant_lines"],
                    "compression_ratio": meta["compression_ratio"],
                    "flags": classify(sample, relevant_lines),
                    "task_preview": task.splitlines()[0][:200],
                    "output_preview": tool_output.splitlines()[:12],
                    "relevant_preview": relevant_lines[:12],
                    "raw_match": raw_index.get(key),
                    "review_status": "pending",
                    "review_decision": None,
                    "review_note": "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {len(manifest)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
