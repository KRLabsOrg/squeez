"""Download the training dataset from HuggingFace."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    if train_path.exists() and eval_path.exists():
        logger.info("Data already downloaded. Delete data/ to re-download.")
        return

    logger.info("Downloading dataset from HuggingFace...")
    ds = load_dataset("KRLabsOrg/tool-output-extraction-swebench")

    for split, path in [("train", train_path), ("eval", eval_path)]:
        logger.info(f"Writing {split} to {path}")
        with open(path, "w") as f:
            for sample in ds[split]:
                f.write(json.dumps(sample) + "\n")

    logger.info(f"Done: {len(ds['train'])} train + {len(ds['eval'])} eval samples")


if __name__ == "__main__":
    main()
