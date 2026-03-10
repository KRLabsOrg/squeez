"""HuggingFace custom model for line-level classification of tool output.

Input format:
    [CLS] task_description [SEP] line_1 [LINE_SEP] line_2 [LINE_SEP] ... line_n [SEP]

Output:
    Token-level logits [seq_len, 2] (0=irrelevant, 1=relevant).
    At inference, token predictions are aggregated per line via max-pool.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from squeez.encoder.chunking import chunk_output_lines, encode_text

logger = logging.getLogger(__name__)

LINE_SEP_TOKEN = "[LINE_SEP]"

# Reserve at least this many tokens for tool output lines
_MIN_LINE_BUDGET = 64


class SqueezEncoderConfig(PretrainedConfig):
    """Configuration for SqueezEncoderForLineClassification."""

    model_type = "squeez-encoder"

    def __init__(
        self,
        base_model_name: str = "jhu-clsp/mmBERT-base",
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


class SqueezEncoderForLineClassification(PreTrainedModel):
    """Token-level binary classifier on top of an encoder (mmBERT/ModernBERT).

    Designed to classify each token in the answer portion as relevant (1)
    or irrelevant (0). At inference time, per-line scores are computed by
    max-pooling token scores between [LINE_SEP] markers.

    The encoder is created from config (no pretrained weights) in __init__.
    Use ``from_encoder_pretrained`` to initialise with pretrained encoder
    weights for training, or ``from_pretrained`` to load a saved checkpoint.
    """

    config_class = SqueezEncoderConfig

    def __init__(self, config: SqueezEncoderConfig):
        super().__init__(config)

        # Build encoder from stored config dict (avoids from_pretrained
        # during HF's meta-device init used by from_pretrained).
        if config.encoder_config is not None:
            enc_dict = dict(config.encoder_config)
            model_type = enc_dict.pop("model_type", None)
            base_cfg = AutoConfig.for_model(model_type, **enc_dict)
        else:
            base_cfg = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
            config.encoder_config = base_cfg.to_dict()

        # Set vocab size in encoder config so embeddings are created at
        # the correct size (avoids resize_token_embeddings on meta device).
        if config.vocab_size is not None:
            base_cfg.vocab_size = config.vocab_size

        self.encoder = AutoModel.from_config(base_cfg, trust_remote_code=True)
        hidden_size = base_cfg.hidden_size
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls, config: SqueezEncoderConfig
    ) -> SqueezEncoderForLineClassification:
        """Create a model and load pretrained encoder weights (for training).

        This is the entry point for *new* training runs.  For loading a
        previously-saved checkpoint, use ``from_pretrained`` instead.
        """
        # Temporarily clear vocab_size so __init__ doesn't resize before
        # we load the pretrained weights.
        saved_vocab_size = config.vocab_size
        config.vocab_size = None

        model = cls(config)

        pretrained = AutoModel.from_pretrained(config.base_model_name, trust_remote_code=True)
        model.encoder.load_state_dict(pretrained.state_dict())
        del pretrained

        # Now resize for the extended tokenizer vocab
        if saved_vocab_size is not None:
            model.encoder.resize_token_embeddings(saved_vocab_size)
        config.vocab_size = saved_vocab_size

        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            flat_labels = labels.view(-1)
            valid = flat_labels != -100
            n_pos = (flat_labels[valid] == 1).sum().float()
            n_neg = (flat_labels[valid] == 0).sum().float()
            if n_pos > 0 and n_neg > 0:
                weight = torch.tensor(
                    [1.0, (n_neg / n_pos).clamp(max=10.0)],
                    device=logits.device,
                )
            else:
                weight = None
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
            loss = loss_fn(logits.view(-1, self.config.num_labels), flat_labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def extract(
        self,
        task: str,
        tool_output: str,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        window_overlap: int = 2,
    ) -> list[str]:
        """High-level inference: return list of relevant line strings.

        Handles sliding-window chunking for long tool outputs that exceed
        the model's max sequence length.
        """
        lines = tool_output.split("\n")
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            return []

        max_len = self.config.max_length
        sep_id = tokenizer.sep_token_id

        # Tokenize task prefix (will be reused for every window)
        task_ids = encode_text(
            tokenizer,
            task,
            truncation=True,
            max_length=max(max_len - 3 - _MIN_LINE_BUDGET, 0),
        )

        # Budget: [CLS] task [SEP] ... lines ... [SEP]
        prefix_len = 1 + len(task_ids) + 1  # CLS + task + SEP
        suffix_len = 1  # final SEP
        budget = max_len - prefix_len - suffix_len

        # Tokenize lines, chunking only pathological long lines.
        line_token_ids, chunk_to_line = chunk_output_lines(
            tokenizer,
            lines,
            max_tokens_per_chunk=max(max_len - 4, 1),
        )
        if not line_token_ids:
            return []

        # Build windows
        windows = self._build_windows(line_token_ids, budget, window_overlap)

        # Per-line score aggregation (max across windows)
        line_scores = [0.0] * len(lines)

        for start_idx, end_idx in windows:
            window_lines = line_token_ids[start_idx:end_idx]
            input_ids, line_sep_positions = self._build_input(task_ids, window_lines, tokenizer)
            attention_mask = torch.ones_like(input_ids)

            scores = self._predict_window(input_ids, attention_mask, line_sep_positions, sep_id)
            for i, score in enumerate(scores):
                chunk_idx = start_idx + i
                line_idx = chunk_to_line[chunk_idx]
                line_scores[line_idx] = max(line_scores[line_idx], score)

        return [line for line, score in zip(lines, line_scores) if score >= threshold]

    def _build_windows(
        self,
        line_token_ids: list[list[int]],
        budget: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Split lines into windows that fit within the token budget.

        Each window is a (start_line_idx, end_line_idx) tuple.
        A [LINE_SEP] token is inserted between lines, so the cost per line
        is len(tokens) + 1 (except the first line in each window).
        """
        n = len(line_token_ids)
        if n == 0:
            return []

        windows: list[tuple[int, int]] = []
        start = 0

        while start < n:
            used = len(line_token_ids[start])  # first line, no LINE_SEP before it
            end = start + 1
            while end < n:
                cost = 1 + len(line_token_ids[end])  # LINE_SEP + line tokens
                if used + cost > budget:
                    break
                used += cost
                end += 1

            # If even a single line doesn't fit, include it anyway (will be truncated)
            if end == start:
                end = start + 1

            windows.append((start, end))
            # Advance with overlap
            start = max(start + 1, end - overlap)

        return windows

    def _build_input(
        self,
        task_ids: list[int],
        window_line_ids: list[list[int]],
        tokenizer: AutoTokenizer,
    ) -> tuple[torch.Tensor, list[int]]:
        """Build input_ids tensor for one window.

        Format: [CLS] task_ids [SEP] line_0 [LINE_SEP] line_1 ... [LINE_SEP] line_n [SEP]

        Returns (input_ids [1, seq_len], line_sep_positions).
        """
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

        # Truncate to max_length
        ids = ids[: self.config.max_length]

        return (
            torch.tensor([ids], dtype=torch.long),
            line_sep_positions,
        )

    @torch.no_grad()
    def _predict_window(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_sep_positions: list[int],
        sep_token_id: int,
    ) -> list[float]:
        """Run forward pass and return per-line relevance scores for one window.

        Aggregation: max-pool the P(relevant) of tokens belonging to each line.
        Line boundaries are determined by [LINE_SEP] positions.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits[0], dim=-1)[:, 1]  # P(relevant)

        seq_len = input_ids.shape[1]

        # Find the first [SEP] after [CLS] — marks end of task description
        ids_list = input_ids[0].tolist()
        first_sep = -1
        for i, t in enumerate(ids_list):
            if i > 0 and t == sep_token_id:
                first_sep = i
                break

        if first_sep < 0:
            return [0.0]

        answer_start = first_sep + 1  # first token of line_0

        # Line boundaries: answer_start ... [LINE_SEP] ... [LINE_SEP] ... final_sep
        # Find the final [SEP] (last token before padding)
        final_sep = seq_len - 1  # last token is [SEP]

        boundaries = [answer_start] + line_sep_positions + [final_sep]
        num_lines = len(boundaries) - 1

        scores: list[float] = []
        for i in range(num_lines):
            start = boundaries[i]
            end = boundaries[i + 1]
            # Skip the LINE_SEP token itself (it's at `start` for i > 0)
            if i > 0 and start < seq_len:
                start += 1  # skip past [LINE_SEP]
            if start >= end or start >= seq_len:
                scores.append(0.0)
                continue
            line_probs = probs[start:end]
            scores.append(float(line_probs.max()))

        return scores
