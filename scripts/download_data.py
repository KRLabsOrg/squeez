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


if __name__ == "__main__":
    main()
