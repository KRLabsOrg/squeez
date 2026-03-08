# Changelog

All notable changes to Squeez are documented here.

## [0.1.2] - 2026-03-08

### Fixed
- **Chat template**: switched from custom `<|system|>`/`<|user|>`/`<|assistant|>` tokens to Qwen ChatML format (`<|im_start|>`/`<|im_end|>`) — critical for correct fine-tuning and inference
- **Evaluation metrics**: replaced line-number-based metrics with span-level exact match, precision/recall/F1, partial overlap, and empty accuracy
- **Dataset metadata**: recomputed `num_relevant_lines`, `num_total_lines`, and `compression_ratio` from actual response content

### Changed
- **Training**: added Unsloth support for memory-efficient LoRA training on Qwen 3.5 (falls back to vanilla transformers if not installed)
- **Training config**: batch size 8, grad accum 4 (effective BS 32), max_length 16384, eval every 100 steps
- **Data splits**: `download_data.py` now creates train/dev/test (dev split from train for checkpoint selection, test held out for final eval)

### Added
- **Data quality**: manually reviewed test split (55/436 corrected), traceback curation on train split (123/7148 corrected)
- **Data quality docs**: new `docs/guide/data-quality.md` documenting the full QA pipeline

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
