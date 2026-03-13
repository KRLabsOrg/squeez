"""Download the training dataset from HuggingFace."""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Download squeez training data from HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"

    if not args.force and train_path.exists() and dev_path.exists() and test_path.exists():
        logger.info("Data already downloaded. Use --force to re-download.")
        return

    logger.info("Downloading dataset from HuggingFace...")
    ds = load_dataset("KRLabsOrg/tool-output-extraction-swebench")

    splits = [
        (ds["train"], train_path, "train"),
        (ds["dev"], dev_path, "dev"),
        (ds["test"], test_path, "test"),
    ]

    for data, path, name in splits:
        logger.info(f"Writing {name} to {path} ({len(data)} samples)")
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    logger.info(f"Done: {len(ds['train'])} train / {len(ds['dev'])} dev / {len(ds['test'])} test")

    # Download encoder and canonical files (uploaded as repo files, not dataset splits)
    from huggingface_hub import hf_hub_download

    repo_id = "KRLabsOrg/tool-output-extraction-swebench"
    extra_files = [
        "encoder_train.jsonl",
        "encoder_dev.jsonl",
        "encoder_test.jsonl",
        "canonical_train.jsonl",
        "canonical_dev.jsonl",
        "canonical_test.jsonl",
    ]
    for filename in extra_files:
        dest = output_dir / filename
        if not args.force and dest.exists():
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="dataset"
            )
            import shutil

            shutil.copy2(downloaded, dest)
            logger.info(f"Downloaded {filename}")
        except Exception as e:
            logger.warning(f"Could not download {filename}: {e}")


if __name__ == "__main__":
    main()
