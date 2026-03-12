"""Phase 7: Assemble distilled samples into canonical/Qwen/encoder splits.

Writes three aligned representations from the same canonical source of truth:
- canonical benchmark files (query + tool_output + gold_spans)
- Qwen SFT files (prompt + XML response)
- encoder files (task + tool_output + relevant_lines)

Synthetic samples are split by tool type: 10% held out for eval.
"""

import json
import logging
import random
from collections import defaultdict

from squeez.data.canonical import canonical_record, extract_relevant_lines
from squeez.data.config import SYSTEM_PROMPT, PipelineConfig

logger = logging.getLogger(__name__)


def _format_prompt(query: str, output: str, background_task: str = "") -> str:
    """Format the input prompt using Qwen ChatML template."""
    if len(query) > 1000:
        query = query[:1000] + "..."
    if len(background_task) > 3000:
        background_task = background_task[:3000] + "..."
    background_block = (
        f"<background_task>\n{background_task}\n</background_task>\n" if background_task else ""
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<query>\n{query}\n</query>\n"
        f"{background_block}"
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
    force_rebuild: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assemble distilled samples into train/dev/test JSONL.

    Args:
        distilled_samples: Samples with distilled outputs from Phase 6
        instances: SWE-bench instances for issue text
        config: Pipeline configuration
        force_rebuild: If True, ignore cached assembled files and rewrite splits

    Returns:
        Tuple of (train_samples, dev_samples, test_samples)
    """
    train_path = config.output_dir / "train.jsonl"
    dev_path = config.output_dir / "dev.jsonl"
    test_path = config.output_dir / "test.jsonl"
    canonical_paths = {
        "train": config.output_dir / "canonical_train.jsonl",
        "dev": config.output_dir / "canonical_dev.jsonl",
        "test": config.output_dir / "canonical_test.jsonl",
    }
    encoder_paths = {
        "train": config.output_dir / "encoder_train.jsonl",
        "dev": config.output_dir / "encoder_dev.jsonl",
        "test": config.output_dir / "encoder_test.jsonl",
    }

    # Return cached
    if not force_rebuild and (
        train_path.exists()
        and dev_path.exists()
        and test_path.exists()
        and all(path.exists() for path in canonical_paths.values())
        and all(path.exists() for path in encoder_paths.values())
    ):
        logger.info("Loading cached assembled samples")
        train = [json.loads(line) for line in open(train_path)]
        dev = [json.loads(line) for line in open(dev_path)]
        test = [json.loads(line) for line in open(test_path)]
        return train, dev, test

    all_assembled = []

    for sample in distilled_samples:
        instance_id = sample["instance_id"]
        query = sample.get("query") or sample.get("task", "")
        tool_output = sample.get("tool_output") or sample.get("output", "")
        background_task = sample.get("background_task") or sample.get("task", "")
        if not query or not tool_output:
            continue
        prompt = _format_prompt(query=query, output=tool_output, background_task=background_task)

        spans = sample.get("gold_spans") or sample.get("spans", [])
        relevant = extract_relevant_lines(tool_output, spans)

        if relevant:
            response = "<relevant_lines>\n" + "\n".join(relevant) + "\n</relevant_lines>"
        else:
            response = "<relevant_lines>\n</relevant_lines>"

        actual_relevant = len(relevant)
        total_lines = len(tool_output.split("\n")) if tool_output else 0

        source = sample.get("source", "swe")
        canonical = canonical_record(
            instance_id=instance_id,
            source=source,
            tool_type=sample["tool_type"],
            query=query,
            background_task=background_task,
            tool_output=tool_output,
            gold_spans=spans,
            command=sample.get("command", ""),
            is_irrelevant=sample.get("is_irrelevant"),
        )

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
        encoder = {
            "task": query,
            "tool_output": tool_output,
            "relevant_lines": relevant,
            "tool_type": sample["tool_type"],
        }
        split_group_id = sample.get("split_group_id", instance_id)
        all_assembled.append((assembled, canonical, encoder, instance_id, source, split_group_id))

    # Split: SWE by repo, synthetic by tool type (10% test, 5% dev)
    train_samples = []
    dev_samples = []
    test_samples = []

    # SWE samples: split by repo
    swe = [
        (qwen_row, canonical_row, encoder_row, iid)
        for qwen_row, canonical_row, encoder_row, iid, src, _group_id in all_assembled
        if src not in ("synthetic", "synthetic_negative")
    ]
    canonical_train: list[dict] = []
    canonical_dev: list[dict] = []
    canonical_test: list[dict] = []
    encoder_train: list[dict] = []
    encoder_dev: list[dict] = []
    encoder_test: list[dict] = []

    for assembled, canonical_row, encoder_row, instance_id in swe:
        repo = _get_repo_from_instance_id(instance_id)
        split = _assign_split(repo)
        if split == "test":
            test_samples.append(assembled)
            canonical_test.append(canonical_row)
            encoder_test.append(encoder_row)
        elif split == "dev":
            dev_samples.append(assembled)
            canonical_dev.append(canonical_row)
            encoder_dev.append(encoder_row)
        else:
            train_samples.append(assembled)
            canonical_train.append(canonical_row)
            encoder_train.append(encoder_row)

    # Synthetic samples: hold out 10% test + 5% dev per tool type.
    # Split positives and negatives separately so we can cap negatives in
    # test/dev (too many negatives distort per-tool benchmark metrics).
    synth_pos = [
        (a, c, e, group_id) for a, c, e, _, src, group_id in all_assembled if src == "synthetic"
    ]
    synth_neg = [
        (a, c, e, group_id)
        for a, c, e, _, src, group_id in all_assembled
        if src == "synthetic_negative"
    ]

    pos_by_tool: dict[str, list[tuple[dict, dict, dict, str]]] = defaultdict(list)
    neg_by_tool: dict[str, list[tuple[dict, dict, dict, str]]] = defaultdict(list)
    for a, c, e, group_id in synth_pos:
        pos_by_tool[a["metadata"]["tool_type"]].append((a, c, e, group_id))
    for a, c, e, group_id in synth_neg:
        neg_by_tool[a["metadata"]["tool_type"]].append((a, c, e, group_id))

    rng = random.Random(42)

    split_by_group: dict[str, str] = {}

    def _append(split: str, a: dict, c: dict, e: dict):
        if split == "test":
            test_samples.append(a)
            canonical_test.append(c)
            encoder_test.append(e)
        elif split == "dev":
            dev_samples.append(a)
            canonical_dev.append(c)
            encoder_dev.append(e)
        else:
            train_samples.append(a)
            canonical_train.append(c)
            encoder_train.append(e)

    # First split positives by output-origin group
    for tool_type, samples in pos_by_tool.items():
        grouped: dict[str, list[tuple[dict, dict, dict, str]]] = defaultdict(list)
        for sample in samples:
            grouped[sample[3]].append(sample)
        groups = list(grouped.items())
        rng.shuffle(groups)
        n_test = max(1, int(len(groups) * 0.10))
        n_dev = max(1, int(len(groups) * 0.05))
        for idx, (group_id, group_samples) in enumerate(groups):
            split = "train"
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_dev:
                split = "dev"
            split_by_group[group_id] = split
            for a, c, e, _ in group_samples:
                _append(split, a, c, e)

    # Then split negatives: cap at ~10% of positives in test/dev per tool,
    # rest goes to train. For tiny buckets, at most 1 negative.
    for tool_type, negs in neg_by_tool.items():
        del tool_type  # grouping is handled via output-origin split groups
        for a, c, e, group_id in negs:
            split = split_by_group.get(group_id, "train")
            _append(split, a, c, e)

    # Write to disk
    for path, samples in [
        (train_path, train_samples),
        (dev_path, dev_samples),
        (test_path, test_samples),
    ]:
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    for path, samples in [
        (canonical_paths["train"], canonical_train),
        (canonical_paths["dev"], canonical_dev),
        (canonical_paths["test"], canonical_test),
        (encoder_paths["train"], encoder_train),
        (encoder_paths["dev"], encoder_dev),
        (encoder_paths["test"], encoder_test),
    ]:
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    logger.info(
        f"Assembled {len(train_samples)} train + {len(dev_samples)} dev + {len(test_samples)} test samples"
    )
    return train_samples, dev_samples, test_samples
