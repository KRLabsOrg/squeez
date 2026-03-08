"""Download the training dataset from HuggingFace."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEV_RATIO = 0.05


def main():
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"

    if train_path.exists() and dev_path.exists() and test_path.exists():
        logger.info("Data already downloaded. Delete data/ to re-download.")
        return

    logger.info("Downloading dataset from HuggingFace...")
    ds = load_dataset("KRLabsOrg/tool-output-extraction-swebench")

    # Split HF train into train + dev (stratified by repo would be ideal,
    # but a random split is fine since eval repos are already held out)
    train_split = ds["train"].train_test_split(test_size=DEV_RATIO, seed=42)

    splits = [
        (train_split["train"], train_path, "train"),
        (train_split["test"], dev_path, "dev"),
        (ds["test"], test_path, "test"),
    ]

    for data, path, name in splits:
        logger.info(f"Writing {name} to {path} ({len(data)} samples)")
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    logger.info(
        f"Done: {len(train_split['train'])} train / "
        f"{len(train_split['test'])} dev / {len(ds['test'])} test"
    )


if __name__ == "__main__":
    main()
