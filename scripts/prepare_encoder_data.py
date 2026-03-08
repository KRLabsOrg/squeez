"""Prepare encoder training data from curated ChatML JSONL.

Parses the curated train/dev/test.jsonl (ChatML format with prompt + response)
into the encoder format: {task, tool_output, relevant_lines, tool_type}.

The ChatML user body has the format:
    Task: {issue_text}\n\n{tool_output}

We split on the first `\\n\\n` to separate task from tool_output, then validate
that relevant_lines from the response appear in the extracted tool_output.

Usage:
    python scripts/prepare_encoder_data.py
    python scripts/prepare_encoder_data.py --data-dir data --output-dir data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_user_body(prompt: str) -> str:
    """Extract the user message body from a ChatML prompt."""
    start = "<|im_start|>user\n"
    end = "<|im_end|>\n<|im_start|>assistant\n"
    if start not in prompt or end not in prompt:
        return ""
    return prompt.split(start, 1)[1].rsplit(end, 1)[0]


def _parse_response(response: str) -> list[str]:
    """Parse relevant_lines from response JSON."""
    try:
        data = json.loads(response)
        lines = data.get("relevant_lines", [])
        if isinstance(lines, list):
            return [str(line) for line in lines]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _lines_found_in_output(relevant_lines: list[str], tool_output: str) -> bool:
    """Check that all relevant lines appear in the tool output."""
    for line in relevant_lines:
        if not line.strip():
            continue
        norm = _normalize(line)
        if norm not in _normalize(tool_output):
            return False
    return True


def _split_task_and_output(user_body: str) -> tuple[str, str]:
    """Split user body into task description and tool output.

    The format is: "Task: {issue_text}\\n\\n{tool_output}"
    We try splitting on the first \\n\\n. If relevant lines don't validate
    against the tool_output portion, we try later \\n\\n boundaries.
    """
    # Remove "Task: " prefix if present
    if user_body.startswith("Task: "):
        user_body = user_body[6:]

    parts = user_body.split("\n\n")
    if len(parts) < 2:
        return user_body, ""

    # Try first split
    task = parts[0]
    tool_output = "\n\n".join(parts[1:])
    return task, tool_output


def convert_sample(sample: dict) -> dict | None:
    """Convert a single ChatML sample to encoder format.

    Returns None if the sample cannot be converted.
    """
    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    metadata = sample.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    user_body = _extract_user_body(prompt)
    if not user_body:
        return None

    relevant_lines = _parse_response(response)
    task, tool_output = _split_task_and_output(user_body)

    if not tool_output:
        return None

    # Validate: check that relevant lines appear in tool_output
    # If not, try later \n\n boundaries
    if relevant_lines and not _lines_found_in_output(relevant_lines, tool_output):
        # Try splitting at later \n\n boundaries
        body_no_prefix = user_body[6:] if user_body.startswith("Task: ") else user_body
        parts = body_no_prefix.split("\n\n")
        found = False
        for split_idx in range(2, min(len(parts), 6)):
            candidate_task = "\n\n".join(parts[:split_idx])
            candidate_output = "\n\n".join(parts[split_idx:])
            if candidate_output and _lines_found_in_output(relevant_lines, candidate_output):
                task = candidate_task
                tool_output = candidate_output
                found = True
                break
        if not found:
            # Use original split — some lines may be in the task portion
            # (edge case from data assembly), log warning
            logger.debug(
                f"Some relevant lines not found in tool_output for "
                f"instance_id={metadata.get('instance_id', '?')}"
            )

    return {
        "task": task,
        "tool_output": tool_output,
        "relevant_lines": relevant_lines,
        "tool_type": metadata.get("tool_type", "unknown"),
    }


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert a ChatML JSONL file to encoder format.

    Returns (converted_count, skipped_count).
    """
    converted = 0
    skipped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            result = convert_sample(sample)
            if result is None:
                skipped += 1
                continue
            fout.write(json.dumps(result) + "\n")
            converted += 1

    return converted, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ChatML JSONL to encoder training format")
    parser.add_argument("--data-dir", default="data", help="Directory with train/dev/test.jsonl")
    parser.add_argument("--output-dir", default="data", help="Directory for encoder_*.jsonl output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"]
    for split in splits:
        input_path = data_dir / f"{split}.jsonl"
        if not input_path.exists():
            logger.warning(f"Skipping {input_path} (not found)")
            continue

        output_path = output_dir / f"encoder_{split}.jsonl"
        converted, skipped = convert_file(input_path, output_path)
        logger.info(f"{split}: {converted} converted, {skipped} skipped → {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
