# Changelog

All notable changes to Squeez are documented here.

## [0.1.1] - 2026-03-08

### Changed
- Made `torch`, `transformers`, `peft`, and `datasets` optional dependencies
  - `pip install squeez` — lightweight install for API-only usage (vLLM, Groq, etc.)
  - `pip install squeez[local]` — adds local inference deps (`torch`, `transformers`, `peft`)
  - `pip install squeez[train]` — adds training deps (`datasets`)
  - `pip install squeez[all]` — everything

## [0.1.0] - 2026-03-07

### Added
- Initial release
- CLI tool: `cat output.txt | squeez "task description"`
- Python API: `ToolOutputExtractor` with vLLM and transformers backends
- Config file support (`squeez.yaml`, env vars, CLI args)
- LoRA fine-tuning pipeline for Qwen 3.5 2B
- SFT dataset with proper label masking
- Evaluation metrics: line-level F1, ROUGE-L, compression ratio
- Full data generation pipeline from SWE-bench
- Dataset download script for HuggingFace
