"""Tests for squeez core functionality."""

from squeez.data.config import SYSTEM_PROMPT
from squeez.inference.extractor import _format_prompt, _load_config


def test_format_prompt_basic():
    prompt = _format_prompt("Fix the bug", "class Foo:\n    pass")
    assert "Fix the bug" in prompt
    assert "class Foo:" in prompt
    assert SYSTEM_PROMPT in prompt
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt


def test_format_prompt_truncates_long_task():
    long_task = "x" * 5000
    prompt = _format_prompt(long_task, "output")
    assert len(long_task) > 3000
    assert "..." in prompt
    # Should be truncated to 3000 + "..."
    task_section = prompt.split("Task: ")[1].split("\n\n")[0]
    assert len(task_section) == 3003  # 3000 + "..."


def test_format_prompt_empty_task():
    prompt = _format_prompt("", "some output")
    assert "Task: \n" in prompt
    assert "some output" in prompt


def test_system_prompt_has_json_format():
    assert "relevant_lines" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_load_config_returns_dict():
    config = _load_config()
    assert isinstance(config, dict)


class TestSampleAssembler:
    def test_get_repo_from_instance_id(self):
        from squeez.data.sample_assembler import _get_repo_from_instance_id

        assert _get_repo_from_instance_id("django__django-11099") == "django__django"
        assert _get_repo_from_instance_id("pydata__xarray-3114") == "pydata__xarray"
        assert _get_repo_from_instance_id("nodash") == "nodash"

    def test_assign_split(self):
        from squeez.data.sample_assembler import _assign_split

        assert _assign_split("django__django") == "train"
        assert _assign_split("pydata__xarray") == "eval"
        assert _assign_split("pallets__flask") == "eval"
        assert _assign_split("scikit-learn__scikit-learn") == "train"

    def test_format_prompt(self):
        from squeez.data.sample_assembler import _format_prompt

        prompt = _format_prompt("Fix this", "output text")
        assert "Fix this" in prompt
        assert "output text" in prompt


class TestDataset:
    def test_collate_fn_padding(self):
        import torch

        from squeez.training.dataset import collate_fn

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 4, 5]),
            },
            {
                "input_ids": torch.tensor([6, 7]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 8]),
            },
        ]
        result = collate_fn(batch, pad_token_id=0)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

        # Second item should be left-padded
        assert result["input_ids"][1][0].item() == 0
        assert result["attention_mask"][1][0].item() == 0
        assert result["labels"][1][0].item() == -100

    def test_collate_fn_no_padding_needed(self):
        import torch

        from squeez.training.dataset import collate_fn

        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([3, 4]),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([7, 8]),
            },
        ]
        result = collate_fn(batch, pad_token_id=0)
        assert result["input_ids"].shape == (2, 2)


class TestEvaluate:
    def test_compute_rouge_l(self):
        from squeez.training.evaluate import compute_rouge_l

        score = compute_rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

        score = compute_rouge_l("the cat", "the dog")
        assert 0 < score < 1

        score = compute_rouge_l("", "something")
        assert score == 0.0

    def test_compute_compression_ratio(self):
        from squeez.training.evaluate import compute_compression_ratio

        ratio = compute_compression_ratio("a\nb\nc\nd\ne", "a\nb")
        assert ratio > 0
        assert ratio < 1

        ratio = compute_compression_ratio("a", "a")
        assert ratio == 0.0

    def test_extract_line_numbers(self):
        from squeez.training.evaluate import extract_line_numbers

        lines = extract_line_numbers("1: foo\n2: bar\n5: baz")
        assert lines == {1, 2, 5}

        lines = extract_line_numbers("no numbers here")
        assert lines == set()

    def test_line_level_metrics(self):
        from squeez.training.evaluate import compute_line_level_metrics

        metrics = compute_line_level_metrics("1: a\n2: b\n3: c", "1: a\n2: b\n3: c")
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        metrics = compute_line_level_metrics("1: a", "1: a\n2: b")
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
