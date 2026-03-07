"""Phase 2: Clone repos and fetch source files at the base commit.

Uses bare clones for efficient storage. Falls back to GitHub API if cloning
fails. All tool execution in Phase 4 runs against these cloned repos.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import requests

from squeez.data.config import PipelineConfig
from squeez.data.swebench_loader import parse_patch_files

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def clone_repo(repo: str, repos_dir: Path) -> Path | None:
    """Clone a repo as bare if not already present.

    Args:
        repo: GitHub repo (e.g. "django/django")
        repos_dir: Directory to store cloned repos

    Returns:
        Path to the bare repo directory, or None on failure.
    """
    repo_dir = repos_dir / repo.replace("/", "__")

    if repo_dir.exists():
        return repo_dir

    logger.info(f"Cloning {repo} (bare)...")
    try:
        result = subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info(f"Cloned {repo} to {repo_dir}")
            return repo_dir
        else:
            logger.warning(f"Clone failed for {repo}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Clone timed out for {repo}")
        return None
    except Exception as e:
        logger.warning(f"Clone error for {repo}: {e}")
        return None


def git_show_file(repo_dir: Path, commit: str, filepath: str) -> str | None:
    """Fetch a file's contents at a specific commit using git show.

    Works on bare repos without checkout.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{filepath}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def git_ls_tree(repo_dir: Path, commit: str, directory: str) -> list[str]:
    """List files in a directory at a specific commit using git ls-tree.

    Works on bare repos. Returns list of file paths.
    """
    try:
        result = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{commit}:{directory}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            files = []
            for name in result.stdout.strip().split("\n"):
                if name:
                    full_path = f"{directory}/{name}" if directory else name
                    files.append(full_path)
            return files
        return []
    except (subprocess.TimeoutExpired, Exception):
        return []


def _get_sibling_files(
    repo_dir: Path, commit: str, file_path: str
) -> list[str]:
    """Get other source files in the same directory using git ls-tree."""
    directory = str(Path(file_path).parent)
    if directory == ".":
        directory = ""

    all_files = git_ls_tree(repo_dir, commit, directory)
    source_exts = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h"}
    return [
        f for f in all_files
        if f != file_path and any(f.endswith(ext) for ext in source_exts)
    ]


def fetch_sources_for_instance(
    instance: dict,
    repo_dir: Path,
    cache_dir: Path,
) -> dict[str, str]:
    """Fetch source files for a single SWE-bench instance from cloned repo.

    Fetches:
    1. All files modified in the patch
    2. Sibling files (for decoy tool calls)

    Returns dict mapping file path to content.
    """
    instance_id = instance["instance_id"]
    cache_path = cache_dir / f"{instance_id.replace('/', '__')}.json"

    # Return cached
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    commit = instance["base_commit"]
    patch_files = parse_patch_files(instance["patch"])

    sources: dict[str, str] = {}

    # Fetch patch files
    for fpath in patch_files:
        content = git_show_file(repo_dir, commit, fpath)
        if content is not None:
            sources[fpath] = content

    # Fetch sibling files for decoys
    sibling_paths: set[str] = set()
    for fpath in patch_files[:5]:
        siblings = _get_sibling_files(repo_dir, commit, fpath)
        for sib in siblings[:6]:
            if sib not in sources:
                sibling_paths.add(sib)

    for sib_path in list(sibling_paths)[:15]:
        content = git_show_file(repo_dir, commit, sib_path)
        if content is not None:
            sources[sib_path] = content

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(sources, f)

    logger.info(f"Fetched {len(sources)} files for {instance_id}")
    return sources


def fetch_all_sources(
    instances: list[dict], config: PipelineConfig
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    """Fetch source files for all instances by cloning repos.

    Returns:
        Tuple of:
        - Dict mapping instance_id to {file_path: content}
        - Dict mapping instance_id to repo_dir Path (for running commands)
    """
    all_sources: dict[str, dict[str, str]] = {}
    repo_dirs: dict[str, Path] = {}
    clone_failed: set[str] = set()

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        repo = instance["repo"]

        logger.info(f"[{i+1}/{len(instances)}] Fetching sources for {instance_id}")

        # Clone repo if not already done
        if repo in clone_failed:
            logger.warning(f"Skipping {instance_id} — repo {repo} failed to clone")
            continue

        repo_dir = clone_repo(repo, config.repos_dir)
        if repo_dir is None:
            clone_failed.add(repo)
            logger.warning(f"Skipping {instance_id} — clone failed for {repo}")
            continue

        repo_dirs[instance_id] = repo_dir
        sources = fetch_sources_for_instance(instance, repo_dir, config.source_cache_dir)
        all_sources[instance_id] = sources

    logger.info(f"Fetched sources for {len(all_sources)} instances ({len(clone_failed)} repos failed)")
    return all_sources, repo_dirs


def number_lines(content: str) -> str:
    """Add line numbers to content."""
    lines = content.split("\n")
    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i}: {line}")
    return "\n".join(numbered)


def truncate_output(content: str, max_lines: int = 500) -> str:
    """Truncate content to a maximum number of lines."""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content
    truncated = lines[:max_lines]
    truncated.append(f"... ({len(lines) - max_lines} more lines omitted)")
    return "\n".join(truncated)
