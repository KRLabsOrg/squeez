# Contributing to Squeez

Thanks for your interest in contributing! This guide will get you up and running.

## Setup

```bash
# Clone the repo
git clone https://github.com/KRLabsOrg/squeez.git
cd squeez

# Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting, with a line length of 100.

```bash
# Check formatting
ruff format --check squeez/ tests/

# Auto-format
ruff format squeez/ tests/

# Lint
ruff check squeez/ tests/

# Lint with auto-fix
ruff check --fix squeez/ tests/
```

Key conventions:
- Use modern type hints (`list[str]`, `dict[str, Any]`, `str | None`)
- Add docstrings to public classes and methods
- Use `logging` instead of `print()` for runtime messages
- Use `pathlib.Path` instead of `os.path`

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
squeez/
  inference/         # Runtime extractor (CLI + Python API)
    extractor.py      # ToolOutputExtractor with vLLM/transformers backends
  training/          # Model training pipeline
    train.py          # LoRA fine-tuning script
    dataset.py        # SFT dataset with label masking
    evaluate.py       # Evaluation metrics
  data/              # Data generation pipeline
    pipeline.py       # Main pipeline orchestrator
    config.py         # Configuration and system prompt
    swebench_loader.py
    source_fetcher.py
    tool_call_generator.py
    tool_call_executor.py
    auto_labeler.py
    llm_distiller.py
    sample_assembler.py
    validator.py
configs/             # YAML configuration files
scripts/             # Utility scripts
tests/               # Pytest test suite
```

## Making changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes.** Keep PRs focused — one feature or fix per PR.

3. **Run lint and tests** before committing:
   ```bash
   ruff format squeez/ tests/
   ruff check squeez/ tests/
   pytest tests/ -v
   ```

4. **Open a pull request** against `main`. CI will run lint and tests automatically.

## What to work on

Good areas for contribution:
- **New tool types** — add support for more tool output formats in the data generation pipeline
- **Model backends** — add new inference backends (e.g. GGUF, TensorRT)
- **Evaluation** — improve metrics or add new evaluation methods
- **Tests** — increase coverage
- **Bug fixes** — check [open issues](https://github.com/KRLabsOrg/squeez/issues)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
