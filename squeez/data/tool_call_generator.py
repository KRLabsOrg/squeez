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


def _focus_phrase(
    identifiers: dict[str, list[str]],
    issue_terms: list[str],
    patch_files: list[str],
) -> str:
    """Pick a concrete focus phrase for a tool-specific extraction query."""
    for bucket in ("functions", "classes", "error_strings", "variables"):
        values = identifiers.get(bucket, [])
        if values:
            return values[0]
    if issue_terms:
        return issue_terms[0]
    if patch_files:
        return Path(patch_files[0]).stem.replace("_", " ")
    return "the relevant bug behavior"


def _build_extraction_query(
    tool_type: str,
    identifiers: dict[str, list[str]],
    issue_terms: list[str],
    patch_files: list[str],
) -> str:
    """Build a short, focused extraction query for one tool output."""
    focus = _focus_phrase(identifiers, issue_terms, patch_files)
    if tool_type == "read_file":
        return f"Find the code block most relevant to how {focus} is handled."
    if tool_type == "grep":
        return f"Find the grep hits most relevant to how {focus} is handled."
    if tool_type == "git_log":
        return f"Find the commit entry most relevant to {focus}."
    if tool_type == "git_diff":
        return f"Find the diff hunk most relevant to how {focus} changes."
    if tool_type == "git_blame":
        return f"Find the blame block most relevant to {focus}."
    if tool_type == "ls":
        return f"Find the directory entries most relevant to {focus}."
    if tool_type in {"test_output", "build_output", "lint_output", "type_check", "coverage"}:
        return f"Find the failure block most relevant to {focus}."
    if tool_type == "python":
        return f"Find the runtime output block most relevant to {focus}."
    if tool_type == "curl":
        return f"Find the API or docs block most relevant to how {focus} is described."
    if tool_type == "pip_install":
        return f"Find the dependency error block most relevant to {focus}."
    return f"Find the evidence block most relevant to {focus}."


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


def _generate_read_file_call(patch_files: list[str], sibling_files: list[str]) -> dict:
    """Generate a read_file tool call spec.

    80% patch file, 20% decoy — ensures most reads are task-relevant.
    Decoy selection prefers files in the same directory or imported by patch files.
    """
    if not patch_files and not sibling_files:
        return None

    if patch_files and sibling_files:
        if random.random() < 0.8:
            target = random.choice(patch_files)
            is_patch_file = True
        else:
            # Prefer related decoys: same directory as a patch file
            patch_dirs = {str(Path(f).parent) for f in patch_files}
            related = [f for f in sibling_files if str(Path(f).parent) in patch_dirs]
            pool = related if related else sibling_files
            target = random.choice(pool)
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


def _generate_grep_call(identifiers: dict[str, list[str]], issue_terms: list[str]) -> dict:
    """Generate a grep/search tool call spec."""
    # Pick a search term from identifiers or issue
    candidates = identifiers.get("functions", []) + identifiers.get("classes", []) + issue_terms[:3]
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


def _generate_curl_url_via_llm(
    instance: dict,
    identifiers: dict[str, list[str]],
    patch_files: list[str],
    llm_client,
    llm_model: str,
) -> str | None:
    """Ask an LLM what URL a debugging agent would curl for this task.

    Returns a URL string, or None if the LLM call fails.
    """
    repo = instance["repo"]
    problem = instance.get("problem_statement", "")[:600]
    classes = identifiers.get("classes", [])[:5]
    functions = identifiers.get("functions", [])[:5]

    prompt = (
        "You are a coding agent debugging this issue in the "
        f"{repo} repository.\n\n"
        f"Issue:\n{problem}\n\n"
        f"Modified files: {', '.join(patch_files[:5])}\n"
        f"Key classes: {', '.join(classes)}\n"
        f"Key functions: {', '.join(functions)}\n\n"
        "What single URL would you curl to fetch information useful for "
        "this task? Choose a REAL, publicly accessible URL such as:\n"
        "- The GitHub API endpoint for this repo's specific issue/PR\n"
        "- An official documentation page for the API or class involved\n"
        "- A PyPI/npm package metadata endpoint\n"
        "- A raw file from the repo on GitHub\n\n"
        "Return ONLY the URL on one line, nothing else."
    )

    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        # Extract URL from response (in case model wraps it in backticks etc.)
        url_match = re.search(r"https?://[^\s\"`'<>]+", text)
        if url_match:
            return url_match.group(0).rstrip(".,;)")
    except Exception as e:
        logger.debug(f"LLM curl URL generation failed: {e}")

    return None


def _github_issue_url(instance: dict) -> str:
    """Build the GitHub API URL for this instance's issue (reliable fallback)."""
    repo = instance["repo"]
    instance_id = instance["instance_id"]
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"https://api.github.com/repos/{repo}/issues/{parts[1]}"
    return f"https://api.github.com/repos/{repo}/issues?per_page=5&state=all"


def _generate_curl_call(
    instance: dict,
    identifiers: dict[str, list[str]],
    patch_files: list[str],
    llm_client=None,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """Generate a curl tool call spec.

    If an LLM client is available, asks it to generate a targeted URL that a
    debugging agent would actually fetch. Falls back to the GitHub issue API
    URL (always exists and is directly task-relevant).
    """
    url = None

    # Primary: LLM-generated targeted URL
    if llm_client is not None:
        url = _generate_curl_url_via_llm(instance, identifiers, patch_files, llm_client, llm_model)

    # Fallback: GitHub issue API (always task-relevant)
    if url is None:
        url = _github_issue_url(instance)

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
    # Build a set of realistic python commands
    commands = []

    # Import and inspect a module from the changed file
    if patch_files:
        module = patch_files[0].replace("/", ".").replace(".py", "")
        commands.append(f'python -c "import {module}; print(dir({module}))"')
        if classes:
            cls = random.choice(classes)
            commands.append(f'python -c "from {module} import {cls}; help({cls})"')
            commands.append(f'python -c "from {module} import {cls}; print({cls}.__mro__)"')
        if functions:
            fn = random.choice(functions)
            commands.append(
                f'python -c "from {module} import {fn}; import inspect; print(inspect.getsource({fn}))"'
            )

    # Reproduce the bug from the issue
    commands.append('python -c "import sys; print(sys.version)"')

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


def _generate_build_call(patch_files: list[str] | None = None) -> dict:
    """Generate a build output tool call spec.

    Diversifies beyond just 'python setup.py build' to include
    editable installs, test collection, and import checks.
    """
    commands = [
        "pip install -e . 2>&1",
        "python -m pytest --collect-only 2>&1",
        "python setup.py build 2>&1",
    ]

    # Add import check for the changed module
    if patch_files:
        module = patch_files[0].replace("/", ".").replace(".py", "")
        commands.append(f'python -c "from {module} import *" 2>&1')

    cmd = random.choice(commands)
    return {
        "tool_type": "build_output",
        "command": cmd,
    }


def _generate_pip_install_call() -> dict:
    """Generate a pip install tool call spec."""
    return {
        "tool_type": "pip_install",
        "command": "pip install -e . 2>&1",
    }


def _generate_type_check_call(patch_files: list[str]) -> dict | None:
    """Generate a mypy type check tool call spec."""
    if not patch_files:
        return None
    target = random.choice(patch_files)
    return {
        "tool_type": "type_check",
        "command": f"mypy {target} --no-error-summary 2>&1",
        "target_file": target,
    }


def _generate_coverage_call(instance: dict, patch_files: list[str]) -> dict | None:
    """Generate a pytest coverage tool call spec."""
    fail_to_pass = instance.get("FAIL_TO_PASS", "")
    test_names = []
    if fail_to_pass:
        try:
            test_names = json.loads(fail_to_pass)
            if isinstance(test_names, str):
                test_names = [test_names]
        except (json.JSONDecodeError, TypeError):
            pass

    if not test_names or not patch_files:
        return None

    test = test_names[0]
    module = patch_files[0].replace("/", ".").replace(".py", "")
    return {
        "tool_type": "coverage",
        "command": f"python -m pytest {test} --cov={module} --cov-report=term-missing 2>&1",
        "test_target": test,
        "cov_module": module,
    }


def generate_tool_calls_for_instance(
    instance: dict,
    available_files: list[str],
    llm_client=None,
    llm_model: str = "gpt-4o-mini",
) -> list[dict]:
    """Generate a set of tool call specs for a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict
        available_files: List of file paths available in source cache
        llm_client: Optional OpenAI-compatible client for LLM-grounded curl URLs
        llm_model: Model name to use for LLM calls

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
        "curl": lambda: _generate_curl_call(
            instance, identifiers, patch_files, llm_client, llm_model
        ),
        "git_diff": lambda: _generate_git_diff_call(patch_files),
        "git_blame": lambda: _generate_git_blame_call(patch_files),
        "ls": lambda: _generate_ls_call(patch_files),
        "lint_output": lambda: _generate_lint_call(patch_files),
        "build_output": lambda: _generate_build_call(patch_files),
        "pip_install": lambda: _generate_pip_install_call(),
        "type_check": lambda: _generate_type_check_call(patch_files),
        "coverage": lambda: _generate_coverage_call(instance, patch_files),
    }

    calls = []
    for tool_type in tool_types:
        gen = generators.get(tool_type)
        if gen:
            call = gen()
            if call:
                call["instance_id"] = instance["instance_id"]
                call["query"] = _build_extraction_query(
                    tool_type,
                    identifiers,
                    issue_terms,
                    patch_files,
                )
                calls.append(call)

    return calls


def generate_all_tool_calls(
    instances: list[dict],
    all_sources: dict[str, dict[str, str]],
    config: PipelineConfig,
) -> list[dict]:
    """Generate tool call specs for all instances.

    If an OpenAI API key is available in config, creates an LLM client for
    generating grounded curl URLs. Otherwise falls back to GitHub API URLs.

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

    # Create LLM client for grounded curl URL generation (optional)
    llm_client = None
    llm_model = "gpt-4o-mini"
    api_key = config.openai_api_key
    if api_key:
        try:
            from openai import OpenAI

            kwargs = {"api_key": api_key}
            if config.distillation_base_url:
                kwargs["base_url"] = config.distillation_base_url
            llm_client = OpenAI(**kwargs)
            llm_model = config.distillation_model
            logger.info("LLM client available — curl URLs will be LLM-grounded")
        except Exception as e:
            logger.warning(f"Could not create LLM client for curl URLs: {e}")
    else:
        logger.info("No API key — curl URLs will use GitHub issue API fallback")

    all_calls = []
    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        available_files = list(all_sources.get(instance_id, {}).keys())
        calls = generate_tool_calls_for_instance(instance, available_files, llm_client, llm_model)
        all_calls.extend(calls)

        if (i + 1) % 100 == 0:
            logger.info(f"[Phase 3] {i + 1}/{len(instances)} instances processed")

    # Write to disk
    with open(output_path, "w") as f:
        for call in all_calls:
            f.write(json.dumps(call) + "\n")

    logger.info(f"Generated {len(all_calls)} tool calls for {len(instances)} instances")
    return all_calls
