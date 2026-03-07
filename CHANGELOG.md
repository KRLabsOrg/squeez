# Changelog

All notable changes to Squeez are documented here.

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
