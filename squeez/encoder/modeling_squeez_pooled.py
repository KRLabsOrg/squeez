"""Standalone PooledLineClassifier for HuggingFace AutoModel loading.

This file is self-contained — no squeez package imports required.
It gets copied to the model output directory so that
``AutoModel.from_pretrained("...", trust_remote_code=True)`` works
without installing squeez.

Usage::

    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        "KRLabsOrg/squeez-pooled-modernbert",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "KRLabsOrg/squeez-pooled-modernbert",
    )

    result = model.process(
        task="Find the traceback that shows the import error",
        tool_output=open("output.log").read(),
        tokenizer=tokenizer,
        threshold=0.5,
    )
    print(result["highlighted_lines"])
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)

LINE_SEP_TOKEN = "[LINE_SEP]"
_MIN_LINE_BUDGET = 64


# ---------------------------------------------------------------------------
# Tokenization helpers (inlined from squeez.encoder.chunking)
# ---------------------------------------------------------------------------


class _TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs) -> dict: ...


def _encode_text(
    tokenizer: _TokenizerLike,
    text: str,
    truncation: bool = False,
    max_length: int | None = None,
) -> list[int]:
    """Tokenize a single text span without special tokens."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=truncation,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    input_ids = encoded["input_ids"]
    return input_ids if isinstance(input_ids, list) else list(input_ids)


def _chunk_output_lines(
    tokenizer: _TokenizerLike,
    output_lines: list[str],
    max_tokens_per_chunk: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize output lines, splitting only oversized lines into chunks."""
    chunk_token_ids: list[list[int]] = []
    chunk_to_line: list[int] = []

    for line_idx, line in enumerate(output_lines):
        token_ids = _encode_text(tokenizer, line)
        if not token_ids:
            continue
        if len(token_ids) <= max_tokens_per_chunk:
            chunk_token_ids.append(token_ids)
            chunk_to_line.append(line_idx)
            continue
        for start in range(0, len(token_ids), max_tokens_per_chunk):
            chunk = token_ids[start : start + max_tokens_per_chunk]
            if chunk:
                chunk_token_ids.append(chunk)
                chunk_to_line.append(line_idx)

    return chunk_token_ids, chunk_to_line


def _build_windows(
    line_token_ids: list[list[int]],
    budget: int,
    overlap: int = 2,
) -> list[tuple[int, int]]:
    """Split lines into windows that fit within the token budget."""
    n = len(line_token_ids)
    if n == 0:
        return []
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
        start = max(start + 1, end - overlap)
    return windows


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

    def _pool_lines(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        line_sep_id: int,
        sep_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool hidden states per line for each sample in the batch."""
        batch_size, seq_len, hidden = hidden_states.shape
        device = hidden_states.device

        all_pooled: list[list[torch.Tensor]] = []
        max_lines = 0

        for b in range(batch_size):
            ids = input_ids[b].tolist()

            first_sep = -1
            for i, t in enumerate(ids):
                if i > 0 and t == sep_token_id:
                    first_sep = i
                    break
            if first_sep < 0:
                all_pooled.append([])
                continue

            final_sep = seq_len - 1
            for i in range(seq_len - 1, first_sep, -1):
                if ids[i] == sep_token_id:
                    final_sep = i
                    break

            sep_positions = []
            for i in range(first_sep + 1, final_sep):
                if ids[i] == line_sep_id:
                    sep_positions.append(i)

            boundaries = [first_sep + 1] + sep_positions + [final_sep]
            line_vectors: list[torch.Tensor] = []

            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        pooled, line_mask = self._pool_lines(hidden, input_ids, line_sep_id, sep_token_id)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

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
    # High-level inference API
    # ------------------------------------------------------------------

    def process(
        self,
        task: str,
        tool_output: str,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        window_overlap: int = 2,
        return_line_probabilities: bool = False,
    ) -> dict:
        """Extract relevant lines from tool output.

        Args:
            task: The extraction query / task description.
            tool_output: Raw tool output text.
            tokenizer: The tokenizer for this model.
            threshold: Probability threshold for marking a line as relevant.
            window_overlap: Number of overlapping lines between windows.
            return_line_probabilities: If True, include per-line probabilities.

        Returns:
            Dictionary with keys:
                - ``highlighted_lines``: list of relevant line strings
                - ``highlighted_indices``: list of 0-based line indices
                - ``line_probabilities`` (optional): list of floats per line
        """
        lines = tool_output.split("\n")
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            result = {"highlighted_lines": [], "highlighted_indices": []}
            if return_line_probabilities:
                result["line_probabilities"] = []
            return result

        max_len = self.config.max_length
        sep_id = tokenizer.sep_token_id
        line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)

        task_ids = _encode_text(
            tokenizer,
            task,
            truncation=True,
            max_length=max(max_len - 3 - _MIN_LINE_BUDGET, 0),
        )

        prefix_len = 1 + len(task_ids) + 1
        suffix_len = 1
        budget = max_len - prefix_len - suffix_len

        line_token_ids, chunk_to_line = _chunk_output_lines(
            tokenizer,
            lines,
            max_tokens_per_chunk=max(max_len - 4, 1),
        )
        if not line_token_ids:
            result = {"highlighted_lines": [], "highlighted_indices": []}
            if return_line_probabilities:
                result["line_probabilities"] = [0.0] * len(lines)
            return result

        windows = _build_windows(line_token_ids, budget, window_overlap)

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
            probs = torch.softmax(out["logits"][0], dim=-1)[:, 1]

            n_window_lines = end_idx - start_idx
            for i in range(min(n_window_lines, probs.shape[0])):
                chunk_idx = start_idx + i
                line_idx = chunk_to_line[chunk_idx]
                line_scores[line_idx] = max(line_scores[line_idx], float(probs[i]))

        highlighted_lines = []
        highlighted_indices = []
        for i, (line, score) in enumerate(zip(lines, line_scores)):
            if score >= threshold:
                highlighted_lines.append(line)
                highlighted_indices.append(i)

        result = {
            "highlighted_lines": highlighted_lines,
            "highlighted_indices": highlighted_indices,
        }
        if return_line_probabilities:
            result["line_probabilities"] = line_scores

        return result

    # Alias for compatibility with squeez internal API
    def extract(
        self,
        task: str,
        tool_output: str,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        window_overlap: int = 2,
    ) -> list[str]:
        """Return list of relevant line strings."""
        result = self.process(task, tool_output, tokenizer, threshold, window_overlap)
        return result["highlighted_lines"]

    def _build_input(
        self,
        task_ids: list[int],
        window_line_ids: list[list[int]],
        tokenizer: AutoTokenizer,
    ) -> tuple[torch.Tensor, list[int]]:
        """Build input_ids for one window."""
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
