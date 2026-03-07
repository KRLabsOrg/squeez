# Squeez

Squeeze verbose LLM agent tool output down to only the relevant lines.

[![PyPI](https://img.shields.io/pypi/v/squeez)](https://pypi.org/project/squeez/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow.svg)](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## The Problem

LLM coding agents waste **80-95% of context tokens** on irrelevant tool output. When an agent reads a 500-line file to find one function, or runs `git log` to find a specific commit, most of the output is noise.

Squeez trains a small (2-3B) generative model to identify and extract only the lines that matter for the task at hand — compressing tool output by ~86% on average.

## Installation

```bash
pip install squeez
```

## Quick Start

### CLI

```bash
# Pipe tool output through squeez
cat output.txt | squeez "Fix the CSRF validation bug"

# Or with a file
squeez "Fix the CSRF bug" --input-file output.txt
```

### Python API

```python
from squeez.inference.extractor import ToolOutputExtractor

# Connects to vLLM server (default: localhost:8000)
extractor = ToolOutputExtractor()

# Or load model locally
extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")

filtered = extractor.extract(
    task="Fix the CSRF validation bug in middleware",
    tool_output=raw_output,
)
print(filtered)  # Only the relevant lines
```

The model returns JSON: `{"relevant_lines": ["line1", "line2", ...]}` and the `extract()` method joins them into filtered text.

### Configuration

By default, squeez connects to a vLLM server. Configure with:

```bash
# Environment variable
export TOE_BASE_URL=http://localhost:8000/v1

# Or pass directly
extractor = ToolOutputExtractor(base_url="http://my-server:8000/v1")
```

## Training

### 1. Download the dataset

```bash
python scripts/download_data.py
```

This pulls the [SWE-bench tool output dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) (7,148 train + 436 eval samples) from HuggingFace.

### 2. Train with LoRA

```bash
python -m squeez.training.train \
    --train-file data/train.jsonl \
    --eval-file data/eval.jsonl
```

Default: Qwen 3.5 2B with LoRA (r=16, alpha=32). See `configs/default.yaml` for all hyperparameters.

### 3. Evaluate

```bash
python -m squeez.training.evaluate \
    --model-path output/squeez_qwen \
    --eval-file data/eval.jsonl
```

## Dataset

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

| | Count |
|---|---|
| Train samples | 7,148 |
| Eval samples | 436 |
| With relevant lines | 3,985 (53%) |
| Empty (not relevant) | 3,599 (47%) |
| Avg compression | 86% |

Built from 2,294 [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) instances with real tool execution (git grep, git blame, pytest, ruff, etc.) against 12 repos. Teacher distillation by gpt-oss-120b on Groq.

### Tool types

| Tool Type | Count |
|---|---|
| read_file | 4,309 |
| git_log | 840 |
| grep | 575 |
| build_output | 380 |
| ls | 376 |
| test_output | 344 |
| python | 310 |
| git_blame | 201 |
| lint_output | 101 |
| curl | 95 |
| git_diff | 53 |

## How It Works

1. **Source**: SWE-bench test split (2,294 real GitHub issues)
2. **Tool calls**: 3-7 synthetic tool calls per instance
3. **Real execution**: All commands run against bare-cloned repos at the correct commit
4. **Teacher distillation**: gpt-oss-120b selects relevant line ranges via JSON spans
5. **Zero-hallucination extraction**: Teacher spans matched against original output — no generated text
6. **Assembly**: Extracted lines formatted as `{"relevant_lines": [...]}` for SFT training

## Data Generation

To regenerate the dataset from scratch:

```bash
python -m squeez.data.pipeline --phase all \
    --output-dir data \
    --github-token $GITHUB_TOKEN \
    --openai-api-key $GROQ_API_KEY \
    --distillation-base-url https://api.groq.com/openai/v1
```

## Citation

```bibtex
@software{kovacs2026squeez,
    title={Squeez: Compressing Tool Output for LLM Coding Agents},
    author={Adam Kovacs},
    year={2026},
    url={https://github.com/KRLabsOrg/squeez}
}
```

Built on top of SWE-bench:

```bibtex
@inproceedings{jimenez2024swebench,
    title={SWE-bench: Can Language Models Resolve Real-world Github Issues?},
    author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```

## License

Apache 2.0
