"""Phase 7: Assemble distilled samples into train/eval JSONL for SFT.

Formats each sample as a prompt/response pair with proper chat template
tokens, and splits by SWE-bench repo to ensure zero overlap.
"""

import json
import logging

from squeez.data.config import SYSTEM_PROMPT, PipelineConfig

logger = logging.getLogger(__name__)


def _format_prompt(issue_text: str, output: str) -> str:
    """Format the input prompt for the extraction model."""
    if len(issue_text) > 3000:
        issue_text = issue_text[:3000] + "..."

    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nTask: {issue_text}\n\n{output}\n<|assistant|>\n"


def _get_repo_from_instance_id(instance_id: str) -> str:
    """Extract repo name from instance_id (e.g., 'django__django-11099' -> 'django__django')."""
    parts = instance_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else instance_id


EVAL_REPOS = {"pydata__xarray", "pallets__flask"}


def _assign_split(repo: str) -> str:
    """Assign a repo to train or eval split."""
    return "eval" if repo in EVAL_REPOS else "train"


def assemble_samples(
    distilled_samples: list[dict],
    instances: list[dict],
    config: PipelineConfig,
) -> tuple[list[dict], list[dict]]:
    """Assemble distilled samples into train/eval JSONL.

    Args:
        distilled_samples: Samples with distilled outputs from Phase 6
        instances: SWE-bench instances for issue text
        config: Pipeline configuration

    Returns:
        Tuple of (train_samples, eval_samples)
    """
    train_path = config.output_dir / "train.jsonl"
    eval_path = config.output_dir / "eval.jsonl"

    # Return cached
    if train_path.exists() and eval_path.exists():
        logger.info("Loading cached assembled samples")
        train = [json.loads(line) for line in open(train_path)]
        eval_ = [json.loads(line) for line in open(eval_path)]
        return train, eval_

    instance_map = {inst["instance_id"]: inst for inst in instances}

    train_samples = []
    eval_samples = []

    for sample in distilled_samples:
        instance_id = sample["instance_id"]
        instance = instance_map.get(instance_id)
        if not instance:
            continue

        issue_text = instance.get("problem_statement", "")
        prompt = _format_prompt(
            issue_text=issue_text,
            output=sample["output"],
        )

        # Response is JSON with the relevant lines
        spans = sample.get("spans", [])
        if not spans:
            # No relevant content — empty list
            response = json.dumps({"relevant_lines": []})
        else:
            distilled = sample["distilled_output"]
            relevant = [
                line
                for line in distilled.split("\n")
                if line.strip() and not line.strip().startswith("...")
            ]
            response = json.dumps({"relevant_lines": relevant})

        assembled = {
            "prompt": prompt,
            "response": response,
            "metadata": {
                "instance_id": instance_id,
                "tool_type": sample["tool_type"],
                "compression_ratio": sample.get("compression_ratio", 0),
                "num_total_lines": sample.get("num_total_lines", 0),
                "num_relevant_lines": sample.get("num_relevant_lines", 0),
            },
        }

        # Split by repo
        repo = _get_repo_from_instance_id(instance_id)
        split = _assign_split(repo)

        if split == "train":
            train_samples.append(assembled)
        else:
            eval_samples.append(assembled)

    # Write to disk
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(eval_path, "w") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Assembled {len(train_samples)} train + {len(eval_samples)} eval samples")
    return train_samples, eval_samples
