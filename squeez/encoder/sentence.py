"""Pooled line classifier: single-pass encoder + line-level mean-pool + linear head.

Architecture:
    1. Encode entire input once: [CLS] task [SEP] line_1 [LINE_SEP] line_2 ... [SEP]
    2. Pool hidden states per line (mean-pool tokens between [LINE_SEP] boundaries)
    3. Classify each pooled line vector → relevant/irrelevant

Same input format as the token-level encoder, but classification happens at the
line level instead of the token level. Much more efficient than per-line sentence
classification since the encoder runs only once.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from squeez.encoder.chunking import chunk_output_lines, encode_text
from squeez.encoder.model import LINE_SEP_TOKEN

logger = logging.getLogger(__name__)

_MIN_LINE_BUDGET = 64


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _match_lines(output_lines: list[str], relevant_lines: list[str]) -> list[bool]:
    relevant_set = {_normalize(r) for r in relevant_lines if r.strip()}
    relevant_norms = [_normalize(r) for r in relevant_lines if r.strip()]

    matched = []
    for line in output_lines:
        norm = _normalize(line)
        if not norm:
            matched.append(False)
            continue
        if norm in relevant_set:
            matched.append(True)
            continue
        found = False
        for ref in relevant_norms:
            if norm in ref or ref in norm:
                found = True
                break
        matched.append(found)
    return matched


# ---------------------------------------------------------------------------
# Config + Model
# ---------------------------------------------------------------------------


class PooledLineConfig(PretrainedConfig):
    """Configuration for PooledLineClassifier."""

    model_type = "squeez-pooled"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-large",
        encoder_config: dict | None = None,
        vocab_size: int | None = None,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        max_length: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.encoder_config = encoder_config
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.max_length = max_length


class PooledLineClassifier(PreTrainedModel):
    """Single-pass encoder with line-level mean-pool classification.

    Input:  [CLS] task [SEP] line_0 [LINE_SEP] line_1 [LINE_SEP] ... line_n [SEP]
    Output: One logit pair per line (0=irrelevant, 1=relevant).

    During forward, the encoder runs once over the full input. Hidden states
    are then mean-pooled per line segment (between [LINE_SEP] boundaries),
    and a linear classifier produces per-line predictions.
    """

    config_class = PooledLineConfig

    def __init__(self, config: PooledLineConfig):
        super().__init__(config)

        if config.encoder_config is not None:
            enc_dict = dict(config.encoder_config)
            model_type = enc_dict.pop("model_type", None)
            base_cfg = AutoConfig.for_model(model_type, **enc_dict)
        else:
            base_cfg = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
            config.encoder_config = base_cfg.to_dict()

        if config.vocab_size is not None:
            base_cfg.vocab_size = config.vocab_size

        self.encoder = AutoModel.from_config(base_cfg, trust_remote_code=True)
        hidden_size = base_cfg.hidden_size
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

    @classmethod
    def from_encoder_pretrained(cls, config: PooledLineConfig) -> PooledLineClassifier:
        """Create model with pretrained encoder weights (for new training runs)."""
        saved_vocab_size = config.vocab_size
        config.vocab_size = None

        model = cls(config)

        pretrained = AutoModel.from_pretrained(config.base_model_name, trust_remote_code=True)
        model.encoder.load_state_dict(pretrained.state_dict())
        del pretrained

        if saved_vocab_size is not None:
            model.encoder.resize_token_embeddings(saved_vocab_size)
        config.vocab_size = saved_vocab_size

        return model

    def _pool_lines(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        line_sep_id: int,
        sep_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool hidden states per line for each sample in the batch.

        Returns:
            pooled: [batch, max_lines, hidden] — pooled line representations
            line_mask: [batch, max_lines] — True for real lines, False for padding
        """
        batch_size, seq_len, hidden = hidden_states.shape
        device = hidden_states.device

        # Find line boundaries per sample
        all_pooled: list[list[torch.Tensor]] = []
        max_lines = 0

        for b in range(batch_size):
            ids = input_ids[b].tolist()

            # Find first SEP (end of task) — lines start after it
            first_sep = -1
            for i, t in enumerate(ids):
                if i > 0 and t == sep_token_id:
                    first_sep = i
                    break
            if first_sep < 0:
                all_pooled.append([])
                continue

            # Collect line boundaries: segments between LINE_SEP tokens
            # Lines region: first_sep+1 ... final_sep-1
            # Find final SEP (end of lines) — last non-pad token
            final_sep = seq_len - 1
            for i in range(seq_len - 1, first_sep, -1):
                if ids[i] == sep_token_id:
                    final_sep = i
                    break

            # Collect LINE_SEP positions within lines region
            sep_positions = []
            for i in range(first_sep + 1, final_sep):
                if ids[i] == line_sep_id:
                    sep_positions.append(i)

            # Build line segments
            boundaries = [first_sep + 1] + sep_positions + [final_sep]
            line_vectors: list[torch.Tensor] = []

            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                # Skip the LINE_SEP token itself
                if i > 0:
                    start += 1
                if start >= end:
                    line_vectors.append(torch.zeros(hidden, device=device))
                    continue
                line_vectors.append(hidden_states[b, start:end].mean(dim=0))

            all_pooled.append(line_vectors)
            max_lines = max(max_lines, len(line_vectors))

        if max_lines == 0:
            max_lines = 1

        # Pad to [batch, max_lines, hidden]
        pooled = torch.zeros(batch_size, max_lines, hidden, device=device)
        line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool, device=device)

        for b, vectors in enumerate(all_pooled):
            for i, vec in enumerate(vectors):
                pooled[b, i] = vec
                line_mask[b, i] = True

        return pooled, line_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,
        line_sep_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        **kwargs,
    ):
        """Forward pass with line-level pooling and classification.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            line_labels: [batch, max_lines] — 0/1 per line, -100 for padding
            line_sep_id: token ID for [LINE_SEP]
            sep_token_id: token ID for [SEP]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Pool per line
        pooled, line_mask = self._pool_lines(hidden, input_ids, line_sep_id, sep_token_id)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch, max_lines, 2]

        loss = None
        if line_labels is not None:
            flat_labels = line_labels.view(-1)
            flat_logits = logits.view(-1, self.config.num_labels)

            valid = flat_labels != -100
            n_pos = (flat_labels[valid] == 1).sum().float()
            n_neg = (flat_labels[valid] == 0).sum().float()
            if n_pos > 0 and n_neg > 0:
                weight = torch.tensor([1.0, (n_neg / n_pos).clamp(max=10.0)], device=logits.device)
            else:
                weight = None
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
            loss = loss_fn(flat_logits, flat_labels)

        return {"loss": loss, "logits": logits, "line_mask": line_mask}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(
        self,
        task: str,
        tool_output: str,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        window_overlap: int = 2,
    ) -> list[str]:
        """High-level inference: return list of relevant line strings."""
        lines = tool_output.split("\n")
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            return []

        max_len = self.config.max_length
        sep_id = tokenizer.sep_token_id
        line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)

        task_ids = encode_text(
            tokenizer,
            task,
            truncation=True,
            max_length=max(max_len - 3 - _MIN_LINE_BUDGET, 0),
        )

        prefix_len = 1 + len(task_ids) + 1
        suffix_len = 1
        budget = max_len - prefix_len - suffix_len

        line_token_ids, chunk_to_line = chunk_output_lines(
            tokenizer,
            lines,
            max_tokens_per_chunk=max(max_len - 4, 1),
        )
        if not line_token_ids:
            return []

        from squeez.encoder.model import SqueezEncoderForLineClassification

        windows = SqueezEncoderForLineClassification._build_windows(
            None,
            line_token_ids,
            budget,
            window_overlap,
        )

        line_scores = [0.0] * len(lines)

        for start_idx, end_idx in windows:
            window_lines = line_token_ids[start_idx:end_idx]
            input_ids, _ = self._build_input(task_ids, window_lines, tokenizer)
            attention_mask = torch.ones_like(input_ids)

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                out = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    line_sep_id=line_sep_id,
                    sep_token_id=sep_id,
                )
            probs = torch.softmax(out["logits"][0], dim=-1)[:, 1]  # P(relevant)

            n_window_lines = end_idx - start_idx
            for i in range(min(n_window_lines, probs.shape[0])):
                chunk_idx = start_idx + i
                line_idx = chunk_to_line[chunk_idx]
                line_scores[line_idx] = max(line_scores[line_idx], float(probs[i]))

        return [line for line, score in zip(lines, line_scores) if score >= threshold]

    def _build_input(
        self,
        task_ids: list[int],
        window_line_ids: list[list[int]],
        tokenizer: AutoTokenizer,
    ) -> tuple[torch.Tensor, list[int]]:
        """Build input_ids for one window (same format as token-level encoder)."""
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)

        ids: list[int] = [cls_id] + task_ids + [sep_id]
        line_sep_positions: list[int] = []

        for i, line_ids in enumerate(window_line_ids):
            if i > 0:
                line_sep_positions.append(len(ids))
                ids.append(line_sep_id)
            ids.extend(line_ids)

        ids.append(sep_id)
        ids = ids[: self.config.max_length]

        return torch.tensor([ids], dtype=torch.long), line_sep_positions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PooledLineDataset(Dataset):
    """Dataset for pooled line classification.

    Same input format as the token-level encoder dataset:
        [CLS] task [SEP] line_0 [LINE_SEP] line_1 ... [SEP]

    But labels are per-line (not per-token):
        line_labels: [n_lines] — 0 or 1 per line
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        max_negative_ratio: float | None = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)
        self.sep_token_id = tokenizer.sep_token_id
        self._max_task_tokens = max_length // 2
        self._max_line_tokens = max(max_length - 4, 1)

        raw_samples: list[dict] = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))

        # Build windows (same windowing as token-level encoder)
        self._windows: list[tuple[list[int], list[list[int]], list[int]]] = []
        n_expanded = 0

        for sample in raw_samples:
            task = sample["task"]
            tool_output = sample["tool_output"]
            relevant_lines = sample.get("relevant_lines", [])

            output_lines = tool_output.split("\n")
            line_labels = _match_lines(output_lines, relevant_lines)

            task_ids = encode_text(
                tokenizer,
                task,
                truncation=True,
                max_length=self._max_task_tokens,
            )

            line_token_ids, chunk_to_line = chunk_output_lines(
                tokenizer,
                output_lines,
                max_tokens_per_chunk=self._max_line_tokens,
            )
            chunk_labels = [1 if line_labels[idx] else 0 for idx in chunk_to_line]

            prefix_len = 1 + len(task_ids) + 1
            suffix_len = 1
            budget = max_length - prefix_len - suffix_len

            windows = self._build_windows(line_token_ids, budget)

            for start, end in windows:
                w_line_ids = line_token_ids[start:end]
                w_labels = chunk_labels[start:end]
                if not any(w_line_ids):
                    continue
                self._windows.append((task_ids, w_line_ids, w_labels))

            if len(windows) > 1:
                n_expanded += len(windows) - 1

        # Downsample negatives
        n_downsampled = 0
        if max_negative_ratio is not None and 0 < max_negative_ratio < 1:
            pos_windows = [w for w in self._windows if any(w[2])]
            neg_windows = [w for w in self._windows if not any(w[2])]
            max_neg = int(len(pos_windows) * max_negative_ratio / (1 - max_negative_ratio))
            if len(neg_windows) > max_neg:
                rng = random.Random(seed)
                rng.shuffle(neg_windows)
                n_downsampled = len(neg_windows) - max_neg
                neg_windows = neg_windows[:max_neg]
            self._windows = pos_windows + neg_windows
            rng = random.Random(seed)
            rng.shuffle(self._windows)

        logger.info(
            f"Loaded {len(raw_samples)} samples → {len(self._windows)} windows "
            f"({n_expanded} extra from sliding, {n_downsampled} downsampled)"
        )

    @staticmethod
    def _build_windows(line_token_ids: list[list[int]], budget: int) -> list[tuple[int, int]]:
        n = len(line_token_ids)
        if n == 0:
            return [(0, 0)]
        windows: list[tuple[int, int]] = []
        start = 0
        while start < n:
            used = len(line_token_ids[start])
            end = start + 1
            while end < n:
                cost = 1 + len(line_token_ids[end])
                if used + cost > budget:
                    break
                used += cost
                end += 1
            if end == start:
                end = start + 1
            windows.append((start, end))
            start = max(start + 1, end - 2)
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task_ids, line_token_ids, line_labels = self._windows[idx]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        # Build input_ids: [CLS] task [SEP] line_0 [LINE_SEP] line_1 ... [SEP]
        input_ids: list[int] = [cls_id] + list(task_ids) + [sep_id]
        for i, line_ids in enumerate(line_token_ids):
            if i > 0:
                input_ids.append(self.line_sep_id)
            input_ids.extend(line_ids)
        input_ids.append(sep_id)

        input_ids = input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # Line labels (padded by collator)
        labels_tensor = torch.tensor(line_labels, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "line_labels": labels_tensor,
        }


def collate_pooled_lines(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collator: pad input_ids and line_labels separately."""
    max_seq_len = max(b["input_ids"].shape[0] for b in batch)
    max_lines = max(b["line_labels"].shape[0] for b in batch)

    input_ids = []
    attention_mask = []
    line_labels = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        n_lines = b["line_labels"].shape[0]

        # Pad sequences
        pad_len = max_seq_len - seq_len
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

        # Pad line labels with -100
        label_pad = max_lines - n_lines
        line_labels.append(
            torch.cat([b["line_labels"], torch.full((label_pad,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "line_labels": torch.stack(line_labels),
    }
