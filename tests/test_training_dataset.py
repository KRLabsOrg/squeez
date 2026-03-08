import json

from squeez.training.dataset import ExtractionSFTDataset


class FailingCallTokenizer:
    eos_token = "<eos>"

    def __call__(self, *args, **kwargs):
        raise AssertionError("__call__ should not be used for text-only training tokenization")

    def encode(self, text, truncation=True, max_length=4096):
        del truncation, max_length
        return [ord(ch) for ch in text]


def test_dataset_uses_encode_instead_of_call(tmp_path):
    sample_path = tmp_path / "train.jsonl"
    sample = {"prompt": "abc", "response": "xy"}
    sample_path.write_text(json.dumps(sample) + "\n")

    dataset = ExtractionSFTDataset(str(sample_path), FailingCallTokenizer(), max_length=128)
    item = dataset[0]

    assert item["input_ids"].tolist() == [ord(ch) for ch in "abcxy<eos>"]
    assert item["attention_mask"].tolist() == [1] * len(item["input_ids"])
    assert item["labels"].tolist()[:3] == [-100, -100, -100]
    assert item["labels"].tolist()[3:] == [ord(ch) for ch in "xy<eos>"]
