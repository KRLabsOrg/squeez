"""Training dataset for the encoder line classifier.

Loads JSONL with {task, tool_output, relevant_lines, tool_type} and creates
token-level labels:
  - task tokens → -100 (ignored in loss)
  - [CLS], [SEP], [LINE_SEP] tokens → -100
  - line tokens → 0 (irrelevant) or 1 (relevant)

Long samples are split into sliding windows so that lines beyond
max_length still receive supervision (matching the inference path).

Line matching uses normalized strip + substring containment, consistent with
the generative model's evaluate.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from squeez.encoder.model import LINE_SEP_TOKEN

logger = logging.getLogger(__name__)

# Number of lines of overlap between consecutive windows
_WINDOW_OVERLAP = 2


def _normalize(s: str) -> str:
    """Strip and collapse whitespace for fuzzy matching."""
    return " ".join(s.split())


def _match_lines(output_lines: list[str], relevant_lines: list[str]) -> list[bool]:
    """Determine which output lines are relevant via fuzzy matching.

    For each output line, check if any relevant_line matches it via:
    1. Exact normalized match
    2. Substring containment (either direction)
    """
    relevant_set = {_normalize(r) for r in relevant_lines if r.strip()}
    relevant_norms = [_normalize(r) for r in relevant_lines if r.strip()]

    matched = []
    for line in output_lines:
        norm = _normalize(line)
        if not norm:
            matched.append(False)
            continue

        # Exact match
        if norm in relevant_set:
            matched.append(True)
            continue

        # Substring match
        found = False
        for ref in relevant_norms:
            if norm in ref or ref in norm:
                found = True
                break
        matched.append(found)

    return matched


class LineClassificationDataset(Dataset):
    """PyTorch dataset for encoder-based line classification.

    Each sample is tokenized into:
        [CLS] task [SEP] line_0 [LINE_SEP] line_1 [LINE_SEP] ... line_n [SEP]

    With labels:
        -100 for CLS, task tokens, SEP, LINE_SEP
        0 or 1 for each line token

    Samples whose tool output exceeds ``max_length`` are split into
    overlapping windows so every line is supervised at least once.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)
        self._max_task_tokens = max_length // 2
        self._max_line_tokens = max(max_length - 4, 1)

        raw_samples: list[dict] = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))

        # Pre-tokenise every line and build a flat index of windows.
        # Each entry in self._windows is (task_ids, line_token_ids, line_labels)
        # covering one window of lines that fits within max_length.
        self._windows: list[tuple[list[int], list[list[int]], list[bool]]] = []
        n_expanded = 0

        for sample in raw_samples:
            task = sample["task"]
            tool_output = sample["tool_output"]
            relevant_lines = sample.get("relevant_lines", [])

            output_lines = tool_output.split("\n")
            line_labels = _match_lines(output_lines, relevant_lines)

            # Tokenize task, cap at half of max_length
            task_ids = tokenizer.encode(
                task,
                add_special_tokens=False,
                truncation=True,
                max_length=self._max_task_tokens,
            )

            # Tokenize each line
            line_token_ids = [
                tokenizer.encode(
                    ln,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self._max_line_tokens,
                )
                for ln in output_lines
            ]

            # overhead = [CLS] + task + [SEP] + ... + [SEP]
            prefix_len = 1 + len(task_ids) + 1
            suffix_len = 1
            budget = max_length - prefix_len - suffix_len

            windows = self._build_windows(line_token_ids, budget)

            for start, end in windows:
                self._windows.append(
                    (
                        task_ids,
                        line_token_ids[start:end],
                        line_labels[start:end],
                    )
                )

            if len(windows) > 1:
                n_expanded += len(windows) - 1

        logger.info(
            f"Loaded {len(raw_samples)} samples from {data_path} → "
            f"{len(self._windows)} windows "
            f"({n_expanded} extra from sliding, max_length={max_length})"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_windows(line_token_ids: list[list[int]], budget: int) -> list[tuple[int, int]]:
        """Split lines into windows that fit within the token budget."""
        n = len(line_token_ids)
        if n == 0:
            return [(0, 0)]

        windows: list[tuple[int, int]] = []
        start = 0

        while start < n:
            used = len(line_token_ids[start])
            end = start + 1
            while end < n:
                cost = 1 + len(line_token_ids[end])  # LINE_SEP + line
                if used + cost > budget:
                    break
                used += cost
                end += 1

            if end == start:
                end = start + 1

            windows.append((start, end))
            start = max(start + 1, end - _WINDOW_OVERLAP)

        return windows

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task_ids, line_token_ids, line_labels = self._windows[idx]
        return self._build_example(task_ids, line_token_ids, line_labels)

    def _build_example(
        self,
        task_ids: list[int],
        line_token_ids: list[list[int]],
        line_labels: list[bool],
    ) -> dict[str, torch.Tensor]:
        """Build input_ids and labels for one window."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        # [CLS] task [SEP]
        input_ids: list[int] = [cls_id]
        labels: list[int] = [-100]

        input_ids.extend(task_ids)
        labels.extend([-100] * len(task_ids))

        input_ids.append(sep_id)
        labels.append(-100)

        # Lines with LINE_SEP delimiters
        for i, (line_ids, is_relevant) in enumerate(zip(line_token_ids, line_labels)):
            if i > 0:
                input_ids.append(self.line_sep_id)
                labels.append(-100)

            label_val = 1 if is_relevant else 0
            input_ids.extend(line_ids)
            labels.extend([label_val] * len(line_ids))

        # Final SEP
        input_ids.append(sep_id)
        labels.append(-100)

        # Safety truncate (shouldn't be needed with windowing, but just in case)
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
