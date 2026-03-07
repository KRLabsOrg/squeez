"""Phase 5: Auto-label tool output lines using patch ground truth.

Uses the patch diff to determine which lines in each tool output
are relevant to the bug/fix. Applies quality filters to reject
samples that are too easy or too hard.
"""

import json
import logging
import re
from pathlib import Path

from squeez.data.config import (
    MAX_RELEVANT_RATIO,
    MIN_RELEVANT_LINES,
    MIN_RELEVANT_RATIO,
    MIN_TOTAL_LINES,
    PipelineConfig,
)
from squeez.data.swebench_loader import parse_patch_files, parse_patch_hunks

logger = logging.getLogger(__name__)


def _find_enclosing_scope(content: str, target_line: int) -> list[int]:
    """Find the enclosing function/class definition lines for a target line.

    Simple heuristic: walk backwards from target_line looking for
    def/class lines with less indentation.
    """
    lines = content.split("\n")
    if target_line < 1 or target_line > len(lines):
        return []

    target_indent = len(lines[target_line - 1]) - len(lines[target_line - 1].lstrip())
    scope_lines = []

    for i in range(target_line - 2, -1, -1):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent < target_indent and (
            stripped.startswith("def ") or stripped.startswith("class ")
        ):
            scope_lines.append(i + 1)  # 1-indexed
            target_indent = indent
            if indent == 0:
                break

    return scope_lines


def _find_imports_for_names(content: str, names: set[str]) -> list[int]:
    """Find import lines that reference any of the given names."""
    import_lines = []
    for i, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            for name in names:
                if name in stripped:
                    import_lines.append(i)
                    break
    return import_lines


def label_read_file(
    output: str, instance: dict, tool_output: dict
) -> dict[int, bool] | None:
    """Label read_file output lines using patch hunks.

    Relevant lines:
    - Lines that appear in patch hunks (modified/added)
    - Enclosing function/class headers
    - Related import lines
    """
    target_file = tool_output.get("command", "")
    hunks = parse_patch_hunks(instance["patch"])

    # Get hunk ranges for this file
    file_hunks = hunks.get(target_file, [])
    if not file_hunks and not tool_output.get("is_patch_file", False):
        # Decoy file — mark all lines as irrelevant
        labels = {}
        for i in range(1, output.count("\n") + 2):
            labels[i] = False
        return labels

    # Find relevant line numbers from hunks
    relevant_lines: set[int] = set()
    for start, end in file_hunks:
        for line_no in range(start, end + 1):
            relevant_lines.add(line_no)

    # Parse the original content (strip line numbers from output)
    content_lines = []
    for line in output.split("\n"):
        match = re.match(r"^\d+: (.*)$", line)
        if match:
            content_lines.append(match.group(1))
        else:
            content_lines.append(line)
    content = "\n".join(content_lines)

    # Add enclosing scope lines
    scope_lines: set[int] = set()
    for line_no in relevant_lines:
        scope_lines.update(_find_enclosing_scope(content, line_no))
    relevant_lines.update(scope_lines)

    # Add related imports
    patch_names: set[str] = set()
    patch_names.update(re.findall(r"def (\w+)", instance["patch"]))
    patch_names.update(re.findall(r"class (\w+)", instance["patch"]))
    import_lines = _find_imports_for_names(content, patch_names)
    relevant_lines.update(import_lines)

    # Build label dict
    total_lines = len(content_lines)
    labels = {}
    for i in range(1, total_lines + 1):
        labels[i] = i in relevant_lines

    return labels


def label_grep(output: str, instance: dict) -> dict[int, bool] | None:
    """Label grep output lines.

    Relevant: matches in patch files at or near patch line ranges.
    """
    patch_files = set(parse_patch_files(instance["patch"]))
    hunks = parse_patch_hunks(instance["patch"])

    labels = {}
    for i, line in enumerate(output.split("\n"), 1):
        # Grep output format: file:line_no: content
        match = re.match(r"^([^:]+):(\d+):", line)
        if match:
            file_path = match.group(1)
            line_no = int(match.group(2))
            is_relevant = False

            if file_path in patch_files:
                # Check if near a patch hunk
                file_hunks = hunks.get(file_path, [])
                for start, end in file_hunks:
                    if start - 10 <= line_no <= end + 10:
                        is_relevant = True
                        break
                # If no hunks found but file is in patch, still somewhat relevant
                if not file_hunks:
                    is_relevant = True

            labels[i] = is_relevant
        else:
            labels[i] = False

    return labels


def label_git_log(output: str, instance: dict) -> dict[int, bool] | None:
    """Label git log output. All lines marked as mildly relevant (simulated)."""
    labels = {}
    lines = output.split("\n")
    identifiers = set(re.findall(r"def (\w+)", instance["patch"]))
    identifiers.update(re.findall(r"class (\w+)", instance["patch"]))

    for i, line in enumerate(lines, 1):
        # Mark commits mentioning relevant identifiers as relevant
        is_relevant = any(ident.lower() in line.lower() for ident in identifiers)
        labels[i] = is_relevant

    return labels


def label_test_output(output: str, instance: dict) -> dict[int, bool] | None:
    """Label test output. FAIL lines and tracebacks are relevant."""
    labels = {}
    in_failure = False

    for i, line in enumerate(output.split("\n"), 1):
        if line.startswith("FAIL:") or line.startswith("ERROR:"):
            in_failure = True
            labels[i] = True
        elif line.startswith("---") and in_failure:
            labels[i] = True
            in_failure = False
        elif in_failure:
            labels[i] = True
        elif "FAILED" in line or "Error" in line:
            labels[i] = True
        else:
            labels[i] = False

    return labels


def label_git_diff(output: str, instance: dict) -> dict[int, bool] | None:
    """Label git diff output. Changed lines (+/-) are relevant."""
    labels = {}
    for i, line in enumerate(output.split("\n"), 1):
        if line.startswith("+") or line.startswith("-"):
            labels[i] = not line.startswith("+++") and not line.startswith("---")
        elif line.startswith("@@"):
            labels[i] = True
        elif line.startswith("diff --git"):
            labels[i] = True
        else:
            labels[i] = False
    return labels


def label_ls(output: str, instance: dict) -> dict[int, bool] | None:
    """Label ls output. Files in the patch are relevant."""
    patch_files = set(parse_patch_files(instance["patch"]))
    patch_names = {Path(f).name for f in patch_files}

    labels = {}
    for i, line in enumerate(output.split("\n"), 1):
        is_relevant = any(name in line for name in patch_names)
        labels[i] = is_relevant
    return labels


def label_generic(output: str, instance: dict) -> dict[int, bool] | None:
    """Generic labeling for lint/blame/build output.

    Marks lines containing patch file names or identifiers as relevant.
    """
    patch_files = set(parse_patch_files(instance["patch"]))
    identifiers = set(re.findall(r"def (\w+)", instance["patch"]))
    identifiers.update(re.findall(r"class (\w+)", instance["patch"]))
    all_markers = patch_files | identifiers

    labels = {}
    for i, line in enumerate(output.split("\n"), 1):
        is_relevant = any(marker in line for marker in all_markers)
        labels[i] = is_relevant
    return labels


def auto_label_output(
    tool_output: dict, instance: dict
) -> dict | None:
    """Auto-label a single tool output using patch ground truth.

    Returns a dict with labels and metadata, or None if quality filters reject it.
    """
    output = tool_output["output"]
    tool_type = tool_output["tool_type"]
    total_lines = len(output.split("\n"))

    # Skip if too short
    if total_lines < MIN_TOTAL_LINES:
        return None

    # Select labeler by tool type
    labelers = {
        "read_file": lambda: label_read_file(output, instance, tool_output),
        "grep": lambda: label_grep(output, instance),
        "git_log": lambda: label_git_log(output, instance),
        "test_output": lambda: label_test_output(output, instance),
        "git_diff": lambda: label_git_diff(output, instance),
        "git_blame": lambda: label_generic(output, instance),
        "ls": lambda: label_ls(output, instance),
        "lint_output": lambda: label_generic(output, instance),
        "build_output": lambda: label_generic(output, instance),
    }

    labeler = labelers.get(tool_type)
    if not labeler:
        return None

    labels = labeler()
    if labels is None:
        return None

    # Compute stats
    n_relevant = sum(1 for v in labels.values() if v)
    relevant_ratio = n_relevant / total_lines if total_lines > 0 else 0

    # Quality filters
    if n_relevant < MIN_RELEVANT_LINES:
        logger.debug(f"Rejected: too few relevant lines ({n_relevant})")
        return None
    if relevant_ratio > MAX_RELEVANT_RATIO:
        logger.debug(f"Rejected: too many relevant lines ({relevant_ratio:.2%})")
        return None
    if relevant_ratio < MIN_RELEVANT_RATIO:
        logger.debug(f"Rejected: too few relevant ratio ({relevant_ratio:.2%})")
        return None

    return {
        "instance_id": tool_output["instance_id"],
        "tool_type": tool_type,
        "command": tool_output.get("command", ""),
        "output": output,
        "labels": {str(k): v for k, v in labels.items()},
        "num_total_lines": total_lines,
        "num_relevant_lines": n_relevant,
        "relevant_ratio": round(relevant_ratio, 4),
    }


def auto_label_all(
    tool_outputs: list[dict],
    instances: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    """Auto-label all tool outputs.

    Args:
        tool_outputs: List of executed tool output dicts
        instances: List of SWE-bench instance dicts
        config: Pipeline config

    Returns:
        List of labeled sample dicts
    """
    output_path = config.output_dir / "auto_labels.jsonl"

    # Return cached
    if output_path.exists():
        logger.info(f"Loading cached auto-labels from {output_path}")
        labels = []
        with open(output_path) as f:
            for line in f:
                labels.append(json.loads(line))
        return labels

    instance_map = {inst["instance_id"]: inst for inst in instances}

    labeled = []
    rejected = 0
    for tool_output in tool_outputs:
        instance_id = tool_output["instance_id"]
        instance = instance_map.get(instance_id)
        if not instance:
            continue

        result = auto_label_output(tool_output, instance)
        if result:
            labeled.append(result)
        else:
            rejected += 1

    # Write to disk
    with open(output_path, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    logger.info(
        f"Auto-labeled {len(labeled)} samples ({rejected} rejected by quality filters)"
    )
    return labeled
