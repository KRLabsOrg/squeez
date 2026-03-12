"""Canonical v3 benchmark helpers.

The canonical benchmark format stores:
- a focused extraction query
- raw tool output
- gold spans over that raw output

Model-specific formats (Qwen XML targets, encoder labels) are derived from this
single source of truth.
"""

from __future__ import annotations

from typing import Any


def normalize_spans(spans: list[dict[str, Any]], total_lines: int) -> list[dict[str, int]]:
    """Clamp, sort, and merge line spans against a raw tool output."""
    if total_lines <= 0 or not spans:
        return []

    cleaned: list[tuple[int, int]] = []
    for span in spans:
        start = max(1, int(span.get("start_line", span.get("start", 1))))
        end = min(total_lines, int(span.get("end_line", span.get("end", start))))
        if start <= end:
            cleaned.append((start, end))

    if not cleaned:
        return []

    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return [{"start_line": start, "end_line": end} for start, end in merged]


def extract_relevant_lines(tool_output: str, spans: list[dict[str, Any]]) -> list[str]:
    """Extract relevant lines from raw tool output using canonical spans."""
    lines = tool_output.split("\n")
    normalized = normalize_spans(spans, len(lines))
    result: list[str] = []
    for span in normalized:
        start = span["start_line"] - 1
        end = span["end_line"]
        result.extend(lines[start:end])
    return result


def extract_relevant_text(tool_output: str, spans: list[dict[str, Any]]) -> str:
    """Extract relevant text from raw tool output using canonical spans."""
    return "\n".join(extract_relevant_lines(tool_output, spans))


def canonical_record(
    *,
    instance_id: str,
    source: str,
    tool_type: str,
    query: str,
    tool_output: str,
    gold_spans: list[dict[str, Any]],
    background_task: str = "",
    command: str = "",
    is_irrelevant: bool | None = None,
) -> dict[str, Any]:
    """Build a canonical v3 record."""
    normalized_spans = normalize_spans(gold_spans, len(tool_output.split("\n")))
    if is_irrelevant is None:
        is_irrelevant = len(normalized_spans) == 0

    return {
        "instance_id": instance_id,
        "source": source,
        "tool_type": tool_type,
        "query": query,
        "background_task": background_task,
        "tool_output": tool_output,
        "gold_spans": normalized_spans,
        "is_irrelevant": bool(is_irrelevant),
        "command": command,
    }
