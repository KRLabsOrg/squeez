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

## Training dependencies

For generative model training (Qwen + LoRA):

```bash
pip install -r requirements-train.txt
```

For encoder model training (mmBERT):

```bash
pip install -r requirements-encoder.txt
```

## Dependencies

Squeez requires Python 3.10+. Base install only needs `openai` and `pyyaml`.

Optional dependency groups:

- `pip install squeez[local]` — `torch`, `transformers`, `peft` for local inference
- `pip install squeez[encoder]` — `torch`, `transformers`, `datasets` for encoder training
- `pip install squeez[train]` — adds `trl`, `unsloth` for generative training
- `pip install squeez[dev]` — adds `pytest`, `ruff`
