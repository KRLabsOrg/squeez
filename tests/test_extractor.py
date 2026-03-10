"""Tests for squeez core functionality."""

from squeez.data.config import SYSTEM_PROMPT
from squeez.inference.extractor import _format_prompt, _load_config


def test_format_prompt_basic():
    prompt = _format_prompt("Fix the bug", "class Foo:\n    pass")
    assert "Fix the bug" in prompt
    assert "class Foo:" in prompt
    assert SYSTEM_PROMPT in prompt
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "<|im_end|>" in prompt


def test_format_prompt_truncates_long_task():
    long_task = "x" * 5000
    prompt = _format_prompt(long_task, "output")
    assert len(long_task) > 3000
    assert "..." in prompt
    task_section = prompt.split("<task>\n", 1)[1].split("\n</task>", 1)[0]
    assert len(task_section) == 3003  # 3000 + "..."


def test_format_prompt_empty_task():
    prompt = _format_prompt("", "some output")
    assert "<task>" not in prompt
    assert "some output" in prompt


def test_system_prompt_has_relevant_lines_format():
    assert "relevant_lines" in SYSTEM_PROMPT
    assert "<relevant_lines>" in SYSTEM_PROMPT


def test_load_config_returns_dict():
    config = _load_config()
    assert isinstance(config, dict)


def test_extract_many_preserves_input_order_for_remote_backend():
    from squeez.inference.extractor import ToolOutputExtractor

    extractor = ToolOutputExtractor.__new__(ToolOutputExtractor)
    extractor._backend = "vllm"

    calls = []

    def fake_extract(task, tool_output, max_new_tokens=1024, temperature=0.1):
        del max_new_tokens, temperature
        calls.append((task, tool_output))
        return f"{task}:{tool_output}"

    extractor.extract = fake_extract

    results = extractor.extract_many(
        [("t1", "o1"), ("t2", "o2"), ("t3", "o3")],
        concurrency=3,
    )

    assert sorted(calls) == [("t1", "o1"), ("t2", "o2"), ("t3", "o3")]
    assert results == ["t1:o1", "t2:o2", "t3:o3"]


class TestSampleAssembler:
    def test_get_repo_from_instance_id(self):
        from squeez.data.sample_assembler import _get_repo_from_instance_id

        assert _get_repo_from_instance_id("django__django-11099") == "django__django"
        assert _get_repo_from_instance_id("pydata__xarray-3114") == "pydata__xarray"
        assert _get_repo_from_instance_id("nodash") == "nodash"

    def test_assign_split(self):
        from squeez.data.sample_assembler import _assign_split

        assert _assign_split("django__django") == "train"
        assert _assign_split("pydata__xarray") == "test"
        assert _assign_split("pallets__flask") == "test"
        assert _assign_split("psf__requests") == "dev"
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

    def test_parse_relevant_lines(self):
        from squeez.training.evaluate import _parse_relevant_lines

        lines = _parse_relevant_lines('{"relevant_lines": ["foo", "bar"]}')
        assert lines == ["foo", "bar"]

        lines = _parse_relevant_lines('{"relevant_lines": []}')
        assert lines == []

        # Fallback to raw text
        lines = _parse_relevant_lines("foo\nbar\nbaz")
        assert lines == ["foo", "bar", "baz"]

    def test_span_metrics(self):
        from squeez.training.evaluate import compute_span_metrics

        metrics = compute_span_metrics(["a", "b", "c"], ["a", "b", "c"])
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["exact_match"] == 1.0

        metrics = compute_span_metrics(["a"], ["a", "b"])
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5

        metrics = compute_span_metrics([], [])
        assert metrics["exact_match"] == 1.0

    def test_fuzzy_span_metrics(self):
        from squeez.training.evaluate import compute_fuzzy_span_metrics

        metrics = compute_fuzzy_span_metrics(
            ["ERROR: foo failed at line 12"],
            ["foo failed at line 12"],
            threshold=0.5,
        )
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        metrics = compute_fuzzy_span_metrics(
            ["completely different"], ["foo failed"], threshold=0.5
        )
        assert metrics["f1"] == 0.0

    def test_empty_accuracy(self):
        from squeez.training.evaluate import compute_empty_accuracy

        assert compute_empty_accuracy([], [])["category"] == "true_negative"
        assert compute_empty_accuracy(["a"], ["a"])["category"] == "true_positive"
        assert compute_empty_accuracy(["a"], [])["category"] == "false_positive"
        assert compute_empty_accuracy([], ["a"])["category"] == "false_negative"
