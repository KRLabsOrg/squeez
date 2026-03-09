from squeez.encoder.chunking import chunk_output_lines


class FakeTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    ):
        # Tokenize on whitespace for predictable chunk sizes in tests.
        tokens = [len(part) for part in text.split() if part]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


def test_chunk_output_lines_splits_only_overlong_lines():
    tokenizer = FakeTokenizer()
    lines = [
        "short line",
        "a b c d e f g",
        "tiny",
    ]

    chunks, chunk_to_line = chunk_output_lines(
        tokenizer,
        lines,
        max_tokens_per_chunk=3,
    )

    assert chunks == [
        [5, 4],  # short line
        [1, 1, 1],  # first chunk of long line
        [1, 1, 1],  # second chunk of long line
        [1],  # third chunk of long line
        [4],  # tiny
    ]
    assert chunk_to_line == [0, 1, 1, 1, 2]
