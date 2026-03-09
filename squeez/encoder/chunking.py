"""Helpers for splitting pathological long lines into token chunks.

The encoder task stays line-level by default. Only lines whose tokenized
length exceeds the configured per-line budget are split into chunked
"pseudo-lines". Training assigns the original line label to every chunk.
Inference aggregates chunk scores back to the original line index.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer


def encode_text(
    tokenizer: PreTrainedTokenizer,
    text: str,
    truncation: bool = False,
    max_length: int | None = None,
) -> list[int]:
    """Tokenize a single text span without special tokens or warning spam."""
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


def chunk_output_lines(
    tokenizer: PreTrainedTokenizer,
    output_lines: list[str],
    max_tokens_per_chunk: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize output lines, splitting only oversized lines into chunks.

    Returns:
        chunk_token_ids: token ids for each pseudo-line/chunk
        chunk_to_line: mapping from chunk index back to original line index
    """
    chunk_token_ids: list[list[int]] = []
    chunk_to_line: list[int] = []

    for line_idx, line in enumerate(output_lines):
        token_ids = encode_text(tokenizer, line)
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
