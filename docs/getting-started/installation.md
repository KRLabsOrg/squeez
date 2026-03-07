# Installation

## From PyPI

```bash
pip install squeez
```

## From source

```bash
git clone https://github.com/KRLabsOrg/squeez.git
cd squeez
pip install -e .
```

## Development install

```bash
pip install -e ".[dev]"
```

This adds `pytest` and `ruff` for testing and linting.

## Dependencies

Squeez requires Python 3.10+ and depends on:

- `torch` — model inference and training
- `transformers` — model loading and tokenization
- `peft` — LoRA adapters
- `datasets` — HuggingFace dataset loading
- `openai` — vLLM/API backend
- `pyyaml` — config file parsing
