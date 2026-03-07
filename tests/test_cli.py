"""Tests for the unified squeez CLI."""

from squeez import cli
from squeez.data.pipeline import build_parser as build_pipeline_parser
from squeez.inference.extractor import build_parser as build_extract_parser
from squeez.training.evaluate import build_parser as build_eval_parser
from squeez.training.train import build_parser as build_train_parser


def test_extract_parser_accepts_clear_and_legacy_model_flags():
    parser = build_extract_parser()

    args = parser.parse_args(["find the bug", "--local-model", "./model"])
    assert args.local_model == "./model"

    args = parser.parse_args(["find the bug", "--model-path", "./legacy-model"])
    assert args.local_model == "./legacy-model"

    args = parser.parse_args(["find the bug", "--server-model", "squeez-remote"])
    assert args.server_model == "squeez-remote"

    args = parser.parse_args(["find the bug", "--model-name", "legacy-remote"])
    assert args.server_model == "legacy-remote"


def test_train_parser_accepts_base_model_alias():
    parser = build_train_parser()

    args = parser.parse_args(["--train-file", "train.jsonl", "--base-model", "Qwen/Qwen3.5-2B"])
    assert args.base_model == "Qwen/Qwen3.5-2B"

    args = parser.parse_args(["--train-file", "train.jsonl", "--model", "legacy-model"])
    assert args.base_model == "legacy-model"


def test_eval_parser_accepts_clear_and_legacy_model_flags():
    parser = build_eval_parser()

    args = parser.parse_args(["--extractor-model", "./output/model", "--eval-file", "eval.jsonl"])
    assert args.extractor_model == "./output/model"

    args = parser.parse_args(["--model-path", "./legacy-model", "--eval-file", "eval.jsonl"])
    assert args.extractor_model == "./legacy-model"


def test_pipeline_parser_accepts_teacher_aliases():
    parser = build_pipeline_parser()

    args = parser.parse_args(["--teacher-model", "gpt-5.4"])
    assert args.teacher_model == "gpt-5.4"

    args = parser.parse_args(["--model", "legacy-teacher"])
    assert args.teacher_model == "legacy-teacher"

    args = parser.parse_args(["--teacher-base-url", "http://localhost:8000/v1"])
    assert args.teacher_base_url == "http://localhost:8000/v1"

    args = parser.parse_args(["--base-url", "http://legacy.example/v1"])
    assert args.teacher_base_url == "http://legacy.example/v1"

    args = parser.parse_args(["--teacher-api-key", "secret"])
    assert args.teacher_api_key == "secret"

    args = parser.parse_args(["--api-key", "legacy-secret"])
    assert args.teacher_api_key == "legacy-secret"


def test_cli_defaults_to_extract_mode(monkeypatch):
    called = {}

    def fake_extract(argv):
        called["argv"] = argv
        return 7

    monkeypatch.setattr(cli.extractor, "main", fake_extract)

    assert cli.main(["find the failing test"]) == 7
    assert called["argv"] == ["find the failing test"]


def test_cli_dispatches_named_subcommands(monkeypatch):
    called = {}

    def fake_train(argv):
        called["argv"] = argv
        return 9

    monkeypatch.setitem(cli.SUBCOMMANDS, "train", fake_train)

    assert cli.main(["train", "--train-file", "train.jsonl"]) == 9
    assert called["argv"] == ["--train-file", "train.jsonl"]
