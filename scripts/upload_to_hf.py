"""Upload the assembled dataset to HuggingFace Hub.

Pushes train/dev/test splits + raw source files + DATASET.md.

Usage:
    python scripts/upload_to_hf.py --data-dir data/v3
    python scripts/upload_to_hf.py --data-dir data/v3 --repo KRLabsOrg/tool-output-extraction-swebench
    python scripts/upload_to_hf.py --data-dir data/v3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPO = "KRLabsOrg/tool-output-extraction-swebench"


def upload(data_dir: Path, repo_id: str, dry_run: bool = False) -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"

    for p in [train_path, dev_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run the assembly pipeline first.")

    def load_jsonl(path: Path) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    train = load_jsonl(train_path)
    dev = load_jsonl(dev_path)
    test = load_jsonl(test_path)

    logger.info(f"Loaded: {len(train)} train / {len(dev)} dev / {len(test)} test")

    if dry_run:
        logger.info("[DRY RUN] Would upload to %s", repo_id)
        return

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "dev": Dataset.from_list(dev),
            "test": Dataset.from_list(test),
        }
    )

    logger.info(f"Pushing to {repo_id}...")
    ds.push_to_hub(repo_id, private=False)
    logger.info("Dataset pushed.")

    # Upload raw source files and documentation
    api = HfApi()
    extra_files = []

    # Canonical and encoder splits
    for prefix in ["canonical", "encoder"]:
        for split in ["train", "dev", "test"]:
            p = data_dir / f"{prefix}_{split}.jsonl"
            if p.exists():
                extra_files.append((p, f"{prefix}_{split}.jsonl"))

    # Raw distilled files
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("*.jsonl")):
            extra_files.append((f, f"raw/{f.name}"))

    # DATASET.md as README
    dataset_md = data_dir / "DATASET.md"
    if dataset_md.exists():
        extra_files.append((dataset_md, "README.md"))

    for local_path, repo_path in extra_files:
        logger.info(f"  Uploading {repo_path} ({local_path.stat().st_size / 1024:.0f} KB)")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    logger.info(f"Done. https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload squeez dataset to HuggingFace Hub")
    parser.add_argument("--data-dir", type=Path, default=Path("data/v3"))
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--dry-run", action="store_true", help="Print stats without uploading")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    upload(args.data_dir, args.repo, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
