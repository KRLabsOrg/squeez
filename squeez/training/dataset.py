"""SFT dataset for tool output extraction training.

Formats prompt/response pairs with proper label masking so the model
only trains on generating the response (filtered output), not the prompt.
"""

import json
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ExtractionSFTDataset(Dataset):
    """SFT dataset for tool output extraction.

    Loads JSONL files with 'prompt' and 'response' fields,
    tokenizes them, and masks prompt tokens in labels.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        response = sample["response"]

        # Tokenize full sequence: prompt + response + eos
        full_text = prompt + response + self.tokenizer.eos_token
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Tokenize prompt alone to find where response starts
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoding["input_ids"])

        # Create labels: mask prompt tokens with -100
        input_ids = encoding["input_ids"]
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int | None = 0) -> dict[str, torch.Tensor]:
    """Collate function with left-padding for causal LM training."""
    if pad_token_id is None:
        pad_token_id = 0
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), item["input_ids"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]])
        )
        labels.append(torch.cat([torch.full((pad_len,), -100, dtype=torch.long), item["labels"]]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
