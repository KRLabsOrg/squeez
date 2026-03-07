"""Phase 1: Load SWE-bench instances from HuggingFace."""

import json
import logging
from pathlib import Path

from squeez.data.config import PipelineConfig

logger = logging.getLogger(__name__)


def load_swebench_instances(config: PipelineConfig) -> list[dict]:
    """Load SWE-bench instances from HuggingFace datasets.

    Each instance contains:
        - instance_id: unique identifier (e.g., "django__django-11099")
        - repo: repository name (e.g., "django/django")
        - base_commit: commit SHA before the fix
        - patch: the gold patch that fixes the issue
        - problem_statement: the issue description
        - hints_text: optional hints
        - test_patch: test patch for validation
        - FAIL_TO_PASS: tests that should pass after fix
        - PASS_TO_PASS: tests that should continue passing
        - version: version string
    """
    output_path = config.output_dir / "swebench_instances.json"

    # Return cached if exists
    if output_path.exists():
        logger.info(f"Loading cached instances from {output_path}")
        with open(output_path) as f:
            instances = json.load(f)
        if config.max_instances:
            instances = instances[: config.max_instances]
        logger.info(f"Loaded {len(instances)} instances")
        return instances

    # Load from HuggingFace
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    all_instances = []
    for split in config.splits:
        logger.info(f"Loading SWE-bench split: {split}")
        ds = load_dataset(config.swebench_dataset, split=split)
        for row in ds:
            instance = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "patch": row["patch"],
                "problem_statement": row["problem_statement"],
                "hints_text": row.get("hints_text", ""),
                "test_patch": row.get("test_patch", ""),
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
                "version": row.get("version", ""),
            }
            all_instances.append(instance)

    logger.info(f"Loaded {len(all_instances)} total instances from {config.splits}")

    # Cache to disk
    with open(output_path, "w") as f:
        json.dump(all_instances, f, indent=2)
    logger.info(f"Cached instances to {output_path}")

    if config.max_instances:
        all_instances = all_instances[: config.max_instances]

    return all_instances


def parse_patch_files(patch: str) -> list[str]:
    """Extract file paths modified in a unified diff patch."""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # Format: diff --git a/path/to/file b/path/to/file
            parts = line.split(" b/")
            if len(parts) == 2:
                files.append(parts[1].strip())
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
            if path != "/dev/null":
                files.append(path)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def parse_patch_hunks(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff to extract modified line ranges per file.

    Returns dict mapping file path to list of (start_line, end_line) tuples
    for the post-patch (new) line numbers.
    """
    hunks: dict[str, list[tuple[int, int]]] = {}
    current_file = None

    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in hunks:
                hunks[current_file] = []
        elif line.startswith("@@ ") and current_file:
            # Parse @@ -old_start,old_count +new_start,new_count @@
            try:
                parts = line.split("@@")[1].strip()
                new_part = parts.split("+")[1].split(" ")[0]
                if "," in new_part:
                    start, count = new_part.split(",")
                    start, count = int(start), int(count)
                else:
                    start = int(new_part)
                    count = 1
                hunks[current_file].append((start, start + count - 1))
            except (IndexError, ValueError):
                continue

    return hunks
