"""Phase 7: Assemble distilled samples into train/eval JSONL for SFT.

Formats each sample as a prompt/response pair with proper chat template
tokens, and splits by SWE-bench repo to ensure zero overlap.

Synthetic samples are split by tool type: 10% held out for eval.
"""

import json
import logging
import random
from collections import defaultdict

from squeez.data.config import SYSTEM_PROMPT, PipelineConfig

logger = logging.getLogger(__name__)


def _format_prompt(issue_text: str, output: str) -> str:
    """Format the input prompt using Qwen ChatML template."""
    if len(issue_text) > 3000:
        issue_text = issue_text[:3000] + "..."

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<task>\n{issue_text}\n</task>\n"
        f"<tool_output>\n{output}\n</tool_output><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _get_repo_from_instance_id(instance_id: str) -> str:
    """Extract repo name from instance_id (e.g., 'django__django-11099' -> 'django__django')."""
    parts = instance_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else instance_id


TEST_REPOS = {"pydata__xarray", "pallets__flask"}
DEV_REPOS = {"psf__requests"}


def _assign_split(repo: str) -> str:
    """Assign a repo to train/dev/test split."""
    if repo in TEST_REPOS:
        return "test"
    if repo in DEV_REPOS:
        return "dev"
    return "train"


def assemble_samples(
    distilled_samples: list[dict],
    instances: list[dict],
    config: PipelineConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assemble distilled samples into train/dev/test JSONL.

    Args:
        distilled_samples: Samples with distilled outputs from Phase 6
        instances: SWE-bench instances for issue text
        config: Pipeline configuration

    Returns:
        Tuple of (train_samples, dev_samples, test_samples)
    """
    train_path = config.output_dir / "train.jsonl"
    dev_path = config.output_dir / "dev.jsonl"
    test_path = config.output_dir / "test.jsonl"

    # Return cached
    if train_path.exists() and dev_path.exists() and test_path.exists():
        logger.info("Loading cached assembled samples")
        train = [json.loads(line) for line in open(train_path)]
        dev = [json.loads(line) for line in open(dev_path)]
        test = [json.loads(line) for line in open(test_path)]
        return train, dev, test

    all_assembled = []

    for sample in distilled_samples:
        instance_id = sample["instance_id"]
        issue_text = sample.get("task", "")
        if not issue_text:
            continue
        prompt = _format_prompt(
            issue_text=issue_text,
            output=sample["output"],
        )

        # Response is relevant lines inside XML tags
        spans = sample.get("spans", [])
        if not spans:
            relevant = []
        else:
            distilled = sample["distilled_output"]
            relevant = [
                line
                for line in distilled.split("\n")
                if line.strip() and not line.strip().startswith("...")
            ]

        if relevant:
            response = "<relevant_lines>\n" + "\n".join(relevant) + "\n</relevant_lines>"
        else:
            response = "<relevant_lines>\n</relevant_lines>"

        actual_relevant = len(relevant)
        total_lines = len(sample["output"].split("\n")) if sample.get("output") else 0

        source = sample.get("source", "swe")

        assembled = {
            "prompt": prompt,
            "response": response,
            "metadata": {
                "instance_id": instance_id,
                "tool_type": sample["tool_type"],
                "source": source,
                "num_total_lines": total_lines,
                "num_relevant_lines": actual_relevant,
                "compression_ratio": round(1 - actual_relevant / total_lines, 4)
                if total_lines > 0
                else 0,
            },
        }
        all_assembled.append((assembled, instance_id, source))

    # Split: SWE by repo, synthetic by tool type (10% test, 5% dev)
    train_samples = []
    dev_samples = []
    test_samples = []

    # SWE samples: split by repo
    swe = [
        (a, iid) for a, iid, src in all_assembled if src not in ("synthetic", "synthetic_negative")
    ]
    for assembled, instance_id in swe:
        repo = _get_repo_from_instance_id(instance_id)
        split = _assign_split(repo)
        if split == "test":
            test_samples.append(assembled)
        elif split == "dev":
            dev_samples.append(assembled)
        else:
            train_samples.append(assembled)

    # Synthetic samples: hold out 10% test + 5% dev per tool type.
    # Split positives and negatives separately so we can cap negatives in
    # test/dev (too many negatives distort per-tool benchmark metrics).
    synth_pos = [a for a, _, src in all_assembled if src == "synthetic"]
    synth_neg = [a for a, _, src in all_assembled if src == "synthetic_negative"]

    pos_by_tool: dict[str, list[dict]] = defaultdict(list)
    neg_by_tool: dict[str, list[dict]] = defaultdict(list)
    for a in synth_pos:
        pos_by_tool[a["metadata"]["tool_type"]].append(a)
    for a in synth_neg:
        neg_by_tool[a["metadata"]["tool_type"]].append(a)

    rng = random.Random(42)

    # First split positives
    for tool_type, samples in pos_by_tool.items():
        rng.shuffle(samples)
        n_test = max(1, int(len(samples) * 0.10))
        n_dev = max(1, int(len(samples) * 0.05))
        test_samples.extend(samples[:n_test])
        dev_samples.extend(samples[n_test : n_test + n_dev])
        train_samples.extend(samples[n_test + n_dev :])

    # Then split negatives: cap at ~10% of positives in test/dev per tool,
    # rest goes to train. For tiny buckets, at most 1 negative.
    max_neg_ratio = 0.10
    for tool_type, negs in neg_by_tool.items():
        rng.shuffle(negs)
        n_pos_test = sum(
            1
            for s in test_samples
            if s["metadata"]["tool_type"] == tool_type and s["metadata"]["source"] == "synthetic"
        )
        n_pos_dev = sum(
            1
            for s in dev_samples
            if s["metadata"]["tool_type"] == tool_type and s["metadata"]["source"] == "synthetic"
        )
        max_neg_test = (
            max(0, min(1, int(n_pos_test * max_neg_ratio / (1 - max_neg_ratio))))
            if n_pos_test < 10
            else int(n_pos_test * max_neg_ratio / (1 - max_neg_ratio))
        )
        max_neg_dev = (
            max(0, min(1, int(n_pos_dev * max_neg_ratio / (1 - max_neg_ratio))))
            if n_pos_dev < 10
            else int(n_pos_dev * max_neg_ratio / (1 - max_neg_ratio))
        )
        test_samples.extend(negs[:max_neg_test])
        dev_samples.extend(negs[max_neg_test : max_neg_test + max_neg_dev])
        train_samples.extend(negs[max_neg_test + max_neg_dev :])

    # Write to disk
    for path, samples in [
        (train_path, train_samples),
        (dev_path, dev_samples),
        (test_path, test_samples),
    ]:
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    logger.info(
        f"Assembled {len(train_samples)} train + {len(dev_samples)} dev + {len(test_samples)} test samples"
    )
    return train_samples, dev_samples, test_samples
