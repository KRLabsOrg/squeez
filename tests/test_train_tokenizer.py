import json

from squeez.training.train import (
    _load_dataset_for_sft,
    _prepare_text_tokenizer,
)


class FakeTokenizer:
    def __init__(self, eos_token="<EOS_TOKEN>", pad_token=None, chat_template="x <EOS_TOKEN> y"):
        self.eos_token = eos_token
        self.eos_token_id = None
        self.pad_token = pad_token
        self.pad_token_id = None
        self.chat_template = chat_template
        self.unk_token_id = 999

    def convert_tokens_to_ids(self, token):
        mapping = {"<|im_end|>": 42, "<EOS_TOKEN>": 999}
        return mapping.get(token)


def test_prepare_text_tokenizer_replaces_placeholder_eos():
    tokenizer = FakeTokenizer()

    prepared = _prepare_text_tokenizer("Qwen/Qwen3.5-2B", tokenizer)

    assert prepared.eos_token == "<|im_end|>"
    assert prepared.eos_token_id == 42
    assert prepared.pad_token == "<|im_end|>"
    assert prepared.pad_token_id == 42
    assert prepared.chat_template == "x <|im_end|> y"


def test_load_dataset_for_sft_uses_resolved_eos(tmp_path):
    sample_path = tmp_path / "train.jsonl"
    sample_path.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")

    rows = _load_dataset_for_sft(str(sample_path), "<|im_end|>")

    assert rows == [{"text": "pr<|im_end|>"}]
