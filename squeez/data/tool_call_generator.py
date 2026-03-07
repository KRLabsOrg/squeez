"""Phase 3: Generate synthetic tool call specifications.

Per SWE-bench instance, generates 2-5 tool calls with a realistic mix
of tool types (read_file, grep, git_log, test_output, etc.).
"""

import json
import logging
import random
import re
from pathlib import Path

from squeez.data.config import TOOL_WEIGHTS, PipelineConfig
from squeez.data.swebench_loader import parse_patch_files

logger = logging.getLogger(__name__)


def _extract_identifiers_from_patch(patch: str) -> dict[str, list[str]]:
    """Extract function names, class names, and error strings from a patch."""
    functions = re.findall(r"def (\w+)", patch)
    classes = re.findall(r"class (\w+)", patch)
    # Extract strings that look like error messages or identifiers
    error_strings = re.findall(r'["\']([A-Za-z_]\w{5,}(?:\s\w+){0,3})["\']', patch)
    variables = re.findall(r"(\w{3,})\s*=", patch)

    return {
        "functions": list(set(functions)),
        "classes": list(set(classes)),
        "error_strings": list(set(error_strings))[:5],
        "variables": list(set(variables))[:10],
    }


def _extract_identifiers_from_issue(problem_statement: str) -> list[str]:
    """Extract searchable terms from the issue description."""
    # Look for code references (backtick-wrapped)
    code_refs = re.findall(r"`([^`]{3,50})`", problem_statement)
    # Look for error class names
    errors = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", problem_statement)
    return list(set(code_refs + errors))[:10]


def _pick_tool_types(n: int) -> list[str]:
    """Pick n tool types according to the configured weights."""
    tools = list(TOOL_WEIGHTS.keys())
    weights = list(TOOL_WEIGHTS.values())

    # Always include at least one read_file
    selected = ["read_file"]
    n -= 1

    if n > 0:
        remaining = random.choices(tools, weights=weights, k=n)
        selected.extend(remaining)

    random.shuffle(selected)
    return selected


def _generate_read_file_call(
    patch_files: list[str], sibling_files: list[str]
) -> dict:
    """Generate a read_file tool call spec.

    Mix of patch files and decoys — the model must learn to handle both.
    """
    # 40% patch file, 60% decoy — force the model to handle irrelevant files
    all_files = [(f, True) for f in patch_files] + [(f, False) for f in sibling_files]
    if not all_files:
        return None

    if patch_files and sibling_files:
        if random.random() < 0.4:
            target = random.choice(patch_files)
            is_patch_file = True
        else:
            target = random.choice(sibling_files)
            is_patch_file = False
    elif patch_files:
        target = random.choice(patch_files)
        is_patch_file = True
    else:
        target = random.choice(sibling_files)
        is_patch_file = False

    return {
        "tool_type": "read_file",
        "command": target,
        "target_file": target,
        "is_patch_file": is_patch_file,
    }


def _generate_grep_call(
    identifiers: dict[str, list[str]], issue_terms: list[str]
) -> dict:
    """Generate a grep/search tool call spec."""
    # Pick a search term from identifiers or issue
    candidates = (
        identifiers.get("functions", [])
        + identifiers.get("classes", [])
        + issue_terms[:3]
    )
    if not candidates:
        candidates = identifiers.get("variables", ["TODO"])

    term = random.choice(candidates)
    return {
        "tool_type": "grep",
        "command": f"grep -rn '{term}'",
        "search_term": term,
    }


def _generate_git_log_call(patch_files: list[str]) -> dict:
    """Generate a git log tool call spec."""
    if patch_files:
        target = random.choice(patch_files)
        return {
            "tool_type": "git_log",
            "command": f"git log --oneline -20 -- {target}",
            "target_file": target,
        }
    return {
        "tool_type": "git_log",
        "command": "git log --oneline -20",
    }


def _generate_test_output_call(instance: dict) -> dict:
    """Generate a test output tool call spec."""
    fail_to_pass = instance.get("FAIL_TO_PASS", "")
    return {
        "tool_type": "test_output",
        "command": "python -m pytest",
        "fail_to_pass": fail_to_pass,
        "test_patch": instance.get("test_patch", ""),
    }


def _generate_git_diff_call(patch_files: list[str]) -> dict:
    """Generate a git diff tool call spec."""
    if patch_files:
        target = random.choice(patch_files)
        return {
            "tool_type": "git_diff",
            "command": f"git diff HEAD~5 -- {target}",
            "target_file": target,
        }
    return {
        "tool_type": "git_diff",
        "command": "git diff HEAD~5",
    }


def _generate_git_blame_call(patch_files: list[str]) -> dict:
    """Generate a git blame tool call spec."""
    if patch_files:
        target = random.choice(patch_files)
        return {
            "tool_type": "git_blame",
            "command": f"git blame {target}",
            "target_file": target,
        }
    return None


def _generate_ls_call(patch_files: list[str]) -> dict:
    """Generate an ls/find tool call spec."""
    if patch_files:
        directory = str(Path(patch_files[0]).parent)
        return {
            "tool_type": "ls",
            "command": f"ls -la {directory}/",
            "target_dir": directory,
        }
    return {"tool_type": "ls", "command": "ls -la", "target_dir": "."}


def _generate_lint_call(patch_files: list[str]) -> dict:
    """Generate a lint output tool call spec."""
    if patch_files:
        target = random.choice(patch_files)
        return {
            "tool_type": "lint_output",
            "command": f"ruff check {target}",
            "target_file": target,
        }
    return {"tool_type": "lint_output", "command": "ruff check ."}


def _generate_curl_call(
    instance: dict, identifiers: dict[str, list[str]]
) -> dict:
    """Generate a curl tool call spec.

    Simulates an agent fetching documentation, API responses, or package info
    related to the issue. Common patterns:
    - Fetching library docs for a class/function mentioned in the issue
    - Checking PyPI/npm package metadata
    - Fetching GitHub issue/PR comments
    - Fetching API endpoint responses
    """
    repo = instance["repo"]  # e.g. "astropy/astropy"
    org, project = repo.split("/") if "/" in repo else ("org", repo)

    classes = identifiers.get("classes", [])
    functions = identifiers.get("functions", [])

    url_templates = [
        # ReadTheDocs / Sphinx docs
        (
            f"https://{project}.readthedocs.io/en/latest/api/{random.choice(classes)}.html"
            if classes
            else f"https://{project}.readthedocs.io/en/latest/"
        ),
        # GitHub issue/PR
        f"https://api.github.com/repos/{repo}/issues/{random.randint(1000, 9999)}",
        # PyPI package info
        f"https://pypi.org/pypi/{project}/json",
        # GitHub file content API
        (
            f"https://api.github.com/repos/{repo}/contents/{random.choice(functions)}.py"
            if functions
            else f"https://api.github.com/repos/{repo}/contents/"
        ),
        # Stack Overflow search
        (
            f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={random.choice(classes)}&site=stackoverflow"
            if classes
            else f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={project}&site=stackoverflow"
        ),
    ]

    url = random.choice(url_templates)
    return {
        "tool_type": "curl",
        "command": f"curl -s {url}",
        "url": url,
    }


def _generate_python_call(
    instance: dict, identifiers: dict[str, list[str]], patch_files: list[str]
) -> dict:
    """Generate a python command call spec.

    Simulates an agent running Python to inspect modules, check types,
    reproduce bugs, or test small snippets. These are real commands that
    will be executed against the cloned repo.
    """
    classes = identifiers.get("classes", [])
    functions = identifiers.get("functions", [])
    variables = identifiers.get("variables", [])

    # Build a set of realistic python commands
    commands = []

    # Import and inspect a module from the changed file
    if patch_files:
        module = patch_files[0].replace("/", ".").replace(".py", "")
        commands.append(f"python -c \"import {module}; print(dir({module}))\"")
        if classes:
            cls = random.choice(classes)
            commands.append(f"python -c \"from {module} import {cls}; help({cls})\"")
            commands.append(f"python -c \"from {module} import {cls}; print({cls}.__mro__)\"")
        if functions:
            fn = random.choice(functions)
            commands.append(f"python -c \"from {module} import {fn}; import inspect; print(inspect.getsource({fn}))\"")

    # Reproduce the bug from the issue
    commands.append("python -c \"import sys; print(sys.version)\"")

    # Try to run a specific test
    if instance.get("FAIL_TO_PASS"):
        try:
            tests = json.loads(instance["FAIL_TO_PASS"])
            if isinstance(tests, list) and tests:
                test = tests[0]
                commands.append(f"python -m pytest {test} -x --tb=short")
        except (json.JSONDecodeError, TypeError):
            pass

    cmd = random.choice(commands)
    return {
        "tool_type": "python",
        "command": cmd,
    }


def _generate_build_call() -> dict:
    """Generate a build output tool call spec."""
    return {
        "tool_type": "build_output",
        "command": "python setup.py build",
    }


def generate_tool_calls_for_instance(
    instance: dict, available_files: list[str]
) -> list[dict]:
    """Generate a set of tool call specs for a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict
        available_files: List of file paths available in source cache

    Returns:
        List of tool call spec dicts
    """
    patch_files = parse_patch_files(instance["patch"])
    sibling_files = [f for f in available_files if f not in patch_files]

    identifiers = _extract_identifiers_from_patch(instance["patch"])
    issue_terms = _extract_identifiers_from_issue(instance["problem_statement"])

    # Pick number of tool calls
    n_tools = random.randint(3, 7)
    tool_types = _pick_tool_types(n_tools)

    generators = {
        "read_file": lambda: _generate_read_file_call(patch_files, sibling_files),
        "grep": lambda: _generate_grep_call(identifiers, issue_terms),
        "python": lambda: _generate_python_call(instance, identifiers, patch_files),
        "git_log": lambda: _generate_git_log_call(patch_files),
        "test_output": lambda: _generate_test_output_call(instance),
        "curl": lambda: _generate_curl_call(instance, identifiers),
        "git_diff": lambda: _generate_git_diff_call(patch_files),
        "git_blame": lambda: _generate_git_blame_call(patch_files),
        "ls": lambda: _generate_ls_call(patch_files),
        "lint_output": lambda: _generate_lint_call(patch_files),
        "build_output": lambda: _generate_build_call(),
    }

    calls = []
    for tool_type in tool_types:
        gen = generators.get(tool_type)
        if gen:
            call = gen()
            if call:
                call["instance_id"] = instance["instance_id"]
                calls.append(call)

    return calls


def generate_all_tool_calls(
    instances: list[dict],
    all_sources: dict[str, dict[str, str]],
    config: PipelineConfig,
) -> list[dict]:
    """Generate tool call specs for all instances.

    Args:
        instances: List of SWE-bench instance dicts
        all_sources: Dict mapping instance_id to {file_path: content}
        config: Pipeline config

    Returns:
        List of all tool call spec dicts
    """
    output_path = config.output_dir / "tool_calls.jsonl"

    # Return cached
    if output_path.exists():
        logger.info(f"Loading cached tool calls from {output_path}")
        calls = []
        with open(output_path) as f:
            for line in f:
                calls.append(json.loads(line))
        return calls

    all_calls = []
    for instance in instances:
        instance_id = instance["instance_id"]
        available_files = list(all_sources.get(instance_id, {}).keys())
        calls = generate_tool_calls_for_instance(instance, available_files)
        all_calls.extend(calls)

    # Write to disk
    with open(output_path, "w") as f:
        for call in all_calls:
            f.write(json.dumps(call) + "\n")

    logger.info(f"Generated {len(all_calls)} tool calls for {len(instances)} instances")
    return all_calls
