"""Phase 4: Execute real tool calls against cloned repos.

For each tool call spec from Phase 3, runs the actual command on the cloned
repo at the base commit. All outputs are real — no simulation.
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from squeez.data.config import PipelineConfig
from squeez.data.source_fetcher import git_show_file, number_lines, truncate_output

logger = logging.getLogger(__name__)


def _run_cmd(
    cmd: list[str] | str,
    cwd: str,
    timeout: int = 30,
    shell: bool = False,
) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            # Include stderr for error output (tests, build failures, etc.)
            output = output + "\n" + result.stderr if output else result.stderr
        return output if output else None
    except subprocess.TimeoutExpired:
        return f"[Command timed out after {timeout}s]"
    except Exception as e:
        logger.debug(f"Command failed: {e}")
        return None


def _create_worktree(repo_dir: Path, commit: str) -> str | None:
    """Create a temporary worktree at a specific commit.

    Required for commands that need a working tree (pytest, ruff, python, ls).
    Returns the worktree path, or None on failure.
    """
    tmpdir = tempfile.mkdtemp(prefix="toe_worktree_")
    try:
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", tmpdir, commit],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return tmpdir
        logger.warning(f"Worktree creation failed: {result.stderr[:200]}")
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Worktree error: {e}")
        return None


def _remove_worktree(repo_dir: Path, worktree_path: str):
    """Clean up a temporary worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_path],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass


def execute_read_file(call: dict, repo_dir: Path, commit: str, max_lines: int) -> str | None:
    """Read a file at the base commit using git show."""
    target = call.get("target_file", call.get("command", ""))
    content = git_show_file(repo_dir, commit, target)
    if content is None:
        return None
    numbered = number_lines(content)
    return truncate_output(numbered, max_lines)


def execute_grep(call: dict, repo_dir: Path, commit: str, max_lines: int) -> str | None:
    """Run real git grep on the repo at the base commit."""
    term = call.get("search_term", "")
    if not term:
        return None

    output = _run_cmd(
        ["git", "grep", "-n", "--break", term, commit, "--"],
        cwd=str(repo_dir),
        timeout=30,
    )
    if not output:
        return None

    # Strip the commit prefix from output (commit:file:line: match -> file:line: match)
    prefix = f"{commit}:"
    lines = []
    for line in output.split("\n"):
        if line.startswith(prefix):
            lines.append(line[len(prefix) :])
        else:
            lines.append(line)
    output = "\n".join(lines)
    return truncate_output(output, max_lines)


def execute_git_log(call: dict, repo_dir: Path, commit: str, max_lines: int) -> str | None:
    """Run real git log on the repo."""
    target = call.get("target_file", "")
    cmd = ["git", "log", "--oneline", "-20", commit]
    if target:
        cmd.extend(["--", target])

    output = _run_cmd(cmd, cwd=str(repo_dir), timeout=30)
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_git_diff(call: dict, repo_dir: Path, commit: str, max_lines: int) -> str | None:
    """Run real git diff on the repo."""
    target = call.get("target_file", "")
    # Diff between 5 commits back and the base commit
    cmd = ["git", "diff", f"{commit}~5", commit]
    if target:
        cmd.extend(["--", target])

    output = _run_cmd(cmd, cwd=str(repo_dir), timeout=30)
    if not output:
        # Fallback: show just the most recent commit diff
        cmd = ["git", "diff", f"{commit}~1", commit]
        if target:
            cmd.extend(["--", target])
        output = _run_cmd(cmd, cwd=str(repo_dir), timeout=30)

    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_git_blame(call: dict, repo_dir: Path, commit: str, max_lines: int) -> str | None:
    """Run real git blame on the repo."""
    target = call.get("target_file", "")
    if not target:
        return None

    output = _run_cmd(
        ["git", "blame", commit, "--", target],
        cwd=str(repo_dir),
        timeout=30,
    )
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_ls(
    call: dict, repo_dir: Path, commit: str, max_lines: int, worktree: str | None = None
) -> str | None:
    """Run real ls on the repo worktree or use git ls-tree."""
    target_dir = call.get("target_dir", ".")

    if worktree:
        # Real ls on the worktree
        ls_path = Path(worktree) / target_dir
        if ls_path.exists():
            output = _run_cmd(["ls", "-la", str(ls_path)], cwd=worktree, timeout=10)
            if output:
                return truncate_output(output, max_lines)

    # Fallback to git ls-tree for bare repos
    output = _run_cmd(
        ["git", "ls-tree", "-l", f"{commit}:{target_dir}"],
        cwd=str(repo_dir),
        timeout=30,
    )
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_lint_output(
    call: dict, repo_dir: Path, commit: str, max_lines: int, worktree: str | None = None
) -> str | None:
    """Run real ruff check on a file in the worktree."""
    if not worktree:
        return None

    target = call.get("target_file", "")
    if target:
        target_path = Path(worktree) / target
        if not target_path.exists():
            return None
        cmd = f"cd {worktree} && python -m ruff check {target} 2>&1 || true"
    else:
        cmd = f"cd {worktree} && python -m ruff check . 2>&1 || true"

    output = _run_cmd(cmd, cwd=worktree, timeout=30, shell=True)
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_test_output(
    call: dict,
    repo_dir: Path,
    commit: str,
    instance: dict,
    max_lines: int,
    worktree: str | None = None,
) -> str | None:
    """Run real pytest on the worktree.

    Uses FAIL_TO_PASS test names for targeted test execution.
    """
    if not worktree:
        return None

    fail_to_pass = call.get("fail_to_pass", instance.get("FAIL_TO_PASS", ""))
    test_names = []
    if fail_to_pass:
        try:
            test_names = json.loads(fail_to_pass)
            if isinstance(test_names, str):
                test_names = [test_names]
        except (json.JSONDecodeError, TypeError):
            test_names = [t.strip() for t in str(fail_to_pass).split(",") if t.strip()]

    if test_names:
        # Run specific failing tests
        tests = " ".join(test_names[:3])
        cmd = f"cd {worktree} && python -m pytest {tests} --tb=short -q 2>&1 || true"
    else:
        # Run a limited test suite
        cmd = f"cd {worktree} && python -m pytest --tb=short -q --maxfail=5 -x 2>&1 || true"

    output = _run_cmd(cmd, cwd=worktree, timeout=60, shell=True)
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_python(
    call: dict, repo_dir: Path, commit: str, max_lines: int, worktree: str | None = None
) -> str | None:
    """Run a real python command in the worktree."""
    if not worktree:
        return None

    cmd = call.get("command", "")
    if not cmd:
        return None

    # Run in the worktree directory so imports work
    full_cmd = f"cd {worktree} && {cmd} 2>&1 || true"
    output = _run_cmd(full_cmd, cwd=worktree, timeout=30, shell=True)
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_build_output(
    call: dict, repo_dir: Path, commit: str, max_lines: int, worktree: str | None = None
) -> str | None:
    """Run real build command in the worktree."""
    if not worktree:
        return None

    cmd = call.get("command", "python setup.py build")
    full_cmd = f"cd {worktree} && {cmd} 2>&1 || true"
    output = _run_cmd(full_cmd, cwd=worktree, timeout=120, shell=True)
    if not output:
        return None
    return truncate_output(output, max_lines)


def execute_curl(call: dict, max_lines: int) -> str | None:
    """Run real curl command."""
    url = call.get("url", "")
    if not url:
        return None

    output = _run_cmd(
        ["curl", "-s", "-m", "15", url],
        cwd="/tmp",
        timeout=20,
    )
    if not output:
        return None
    return truncate_output(output, max_lines)


# Commands that need a working tree (not just bare repo)
WORKTREE_TOOLS = {"ls", "lint_output", "test_output", "python", "build_output"}


def execute_tool_call(
    call: dict,
    repo_dir: Path,
    commit: str,
    instance: dict,
    max_lines: int = 500,
    worktree: str | None = None,
) -> dict | None:
    """Execute a single tool call and return the result.

    Args:
        call: Tool call spec from Phase 3
        repo_dir: Path to bare-cloned repo
        commit: Base commit SHA
        instance: SWE-bench instance dict
        max_lines: Maximum output lines
        worktree: Path to checked-out worktree (for commands needing working tree)

    Returns:
        Dict with tool call info and output, or None if execution failed.
    """
    tool_type = call["tool_type"]

    if tool_type == "read_file":
        output = execute_read_file(call, repo_dir, commit, max_lines)
    elif tool_type == "grep":
        output = execute_grep(call, repo_dir, commit, max_lines)
    elif tool_type == "git_log":
        output = execute_git_log(call, repo_dir, commit, max_lines)
    elif tool_type == "git_diff":
        output = execute_git_diff(call, repo_dir, commit, max_lines)
    elif tool_type == "git_blame":
        output = execute_git_blame(call, repo_dir, commit, max_lines)
    elif tool_type == "ls":
        output = execute_ls(call, repo_dir, commit, max_lines, worktree)
    elif tool_type == "lint_output":
        output = execute_lint_output(call, repo_dir, commit, max_lines, worktree)
    elif tool_type == "test_output":
        output = execute_test_output(call, repo_dir, commit, instance, max_lines, worktree)
    elif tool_type == "python":
        output = execute_python(call, repo_dir, commit, max_lines, worktree)
    elif tool_type == "build_output":
        output = execute_build_output(call, repo_dir, commit, max_lines, worktree)
    elif tool_type == "curl":
        output = execute_curl(call, max_lines)
    else:
        logger.warning(f"Unknown tool type: {tool_type}")
        return None

    if output is None:
        logger.debug(f"No output for {tool_type} call in {call.get('instance_id')}")
        return None

    return {
        "instance_id": call["instance_id"],
        "tool_type": tool_type,
        "command": call.get("command", ""),
        "output": output,
        "num_lines": len(output.split("\n")),
        "is_patch_file": call.get("is_patch_file", False),
    }


def _checkout_worktree(worktree_path: str, commit: str) -> bool:
    """Checkout a different commit in an existing worktree. Much faster than create+destroy."""
    try:
        result = subprocess.run(
            ["git", "checkout", "--detach", commit],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def execute_all_tool_calls(
    tool_calls: list[dict],
    repo_dirs: dict[str, Path],
    instances: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    """Execute all tool calls against cloned repos.

    Reuses one worktree per repo (just switches commits) instead of
    creating/destroying per instance. Writes results incrementally.

    Args:
        tool_calls: List of tool call specs from Phase 3
        repo_dirs: Dict mapping instance_id to repo directory Path
        instances: List of SWE-bench instance dicts
        config: Pipeline config

    Returns:
        List of executed tool output dicts
    """
    output_path = config.output_dir / "tool_outputs.jsonl"
    instance_map = {inst["instance_id"]: inst for inst in instances}

    # Group tool calls by instance to share worktrees
    calls_by_instance: dict[str, list[dict]] = {}
    for call in tool_calls:
        iid = call["instance_id"]
        calls_by_instance.setdefault(iid, []).append(call)

    # Load any existing partial results (crash recovery)
    results = []
    done_instances: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results.append(r)
                    done_instances.add(r["instance_id"])

        remaining = len(calls_by_instance) - len(done_instances)
        if remaining <= 0:
            logger.info(f"Loading cached tool outputs from {output_path} ({len(results)} results)")
            return results
        logger.info(
            f"Resuming execution: {len(done_instances)} instances done, {remaining} remaining"
        )

    total_instances = len(calls_by_instance)
    done_count = len(done_instances)
    failed_count = 0
    start_time = time.time()

    # Map instance_id -> repo_dir path (string) for worktree reuse
    repo_for_instance: dict[str, Path] = {}
    for iid in calls_by_instance:
        rd = repo_dirs.get(iid)
        if rd:
            repo_for_instance[iid] = rd

    # Group instances by repo so we can reuse one worktree per repo
    from collections import defaultdict

    instances_by_repo: dict[str, list[str]] = defaultdict(list)
    for iid in calls_by_instance:
        if iid in done_instances:
            continue
        rd = repo_for_instance.get(iid)
        if rd:
            instances_by_repo[str(rd)].append(iid)

    remaining_calls = sum(
        len(calls_by_instance[iid]) for iid in calls_by_instance if iid not in done_instances
    )
    logger.info(
        f"Executing {remaining_calls} calls for {total_instances - done_count} instances "
        f"across {len(instances_by_repo)} repos..."
    )

    for repo_path, instance_ids in instances_by_repo.items():
        repo_dir = Path(repo_path)
        repo_name = repo_dir.name

        # Check if any instance in this repo needs a worktree
        needs_worktree = any(
            any(c["tool_type"] in WORKTREE_TOOLS for c in calls_by_instance[iid])
            for iid in instance_ids
        )

        worktree = None
        if needs_worktree:
            # Create ONE worktree for this entire repo
            first_commit = instance_map.get(instance_ids[0], {}).get("base_commit", "HEAD")
            worktree = _create_worktree(repo_dir, first_commit)
            if worktree:
                logger.info(f"Created worktree for {repo_name} ({len(instance_ids)} instances)")
            else:
                logger.warning(f"Could not create worktree for {repo_name}")

        for iid in instance_ids:
            instance = instance_map.get(iid, {})
            commit = instance.get("base_commit", "HEAD")

            # Switch the worktree to this instance's commit (fast checkout)
            if worktree:
                if not _checkout_worktree(worktree, commit):
                    logger.debug(f"Checkout failed for {iid} at {commit[:8]}")

            instance_results = []
            for call in calls_by_instance[iid]:
                result = execute_tool_call(
                    call, repo_dir, commit, instance, config.max_tool_output_lines, worktree
                )
                if result:
                    instance_results.append(result)
                else:
                    failed_count += 1

            # Write this instance's results immediately (append mode)
            if instance_results:
                with open(output_path, "a") as f:
                    for result in instance_results:
                        f.write(json.dumps(result) + "\n")
                results.extend(instance_results)

            done_count += 1
            if done_count % 50 == 0 or done_count == total_instances:
                elapsed = time.time() - start_time
                rate = (done_count - len(done_instances)) / elapsed if elapsed > 0 else 0
                remaining = total_instances - done_count
                eta = remaining / rate if rate > 0 else 0
                logger.info(
                    f"[Phase 4] {done_count}/{total_instances} instances | "
                    f"{len(results)} outputs | {failed_count} failed | "
                    f"{rate:.1f} inst/s | ETA {eta:.0f}s"
                )

        # Clean up worktree after all instances for this repo are done
        if worktree:
            _remove_worktree(repo_dir, worktree)
            logger.info(f"Cleaned up worktree for {repo_name}")

    logger.info(
        f"Executed {len(results)} tool calls ({failed_count} failed) "
        f"in {time.time() - start_time:.1f}s"
    )
    return results
