# Squeez

<p align="center">
  <img src="https://github.com/KRLabsOrg/squeez/blob/main/assets/squeez_mascot.png?raw=true" alt="Squeez Logo" width="300"/>
  <br><em>Squeeze out the juice, leave the pulp behind.</em>
</p>

Squeeze verbose LLM agent tool output down to only the relevant lines.

[![PyPI](https://img.shields.io/pypi/v/squeez)](https://pypi.org/project/squeez/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow.svg)](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## The Problem

LLM coding agents waste **80-95% of context tokens** on irrelevant tool output. When an agent reads a 500-line file to find one function, or runs `git log` to find a specific commit, most of the output is noise.

Squeez trains small models to identify and extract only the lines that matter for the task at hand — compressing tool output by ~86% on average.

Two approaches are available:

- **Generative** (Qwen 3.5 2B + LoRA) — high-quality extraction via JSON generation
- **Encoder** (mmBERT 307M) — fast line-level binary classification, sliding window over long outputs

## Example

Task: *"Fix the CSRF validation bug in the referer check"*

<table>
<tr>
<th>Before — 42 lines, ~1,200 tokens</th>
<th>After — 8 lines, ~150 tokens</th>
</tr>
<tr>
<td>

```python
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
        if referer is None:
            raise RejectRequest('No referer')
        good_referer = request.get_host()
        if not same_origin(referer, good_referer):
            raise RejectRequest('Bad referer')

    def process_view(self, request, callback, ...):
        if getattr(request, 'csrf_processing_done', False):
            return None
        csrf_token = request.META.get('CSRF_COOKIE')
        if csrf_token is None:
            return self._reject(request, 'No CSRF cookie')
        return self._accept(request)

class SessionMiddleware(MiddlewareMixin):
    def process_request(self, request):
        session_key = request.COOKIES.get(...)
        request.session = self.SessionStore(session_key)

    def process_response(self, request, response):
        if request.session.modified:
            request.session.save()
        return response

class CommonMiddleware(MiddlewareMixin):
    def process_request(self, request):
        host = request.get_host()
        if settings.PREPEND_WWW and ...:
            return redirect(...)

    def process_response(self, request, response):
        if settings.USE_ETAGS:
            response['ETag'] = hashlib.md5(...)
        return response

class SecurityMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if settings.SECURE_SSL_REDIRECT and ...:
            return redirect(...)
```

</td>
<td>

```python
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
        if referer is None:
            raise RejectRequest('No referer')
        good_referer = request.get_host()
        if not same_origin(referer, good_referer):
            raise RejectRequest('Bad referer')
```

**87% compression** — only the CSRF referer logic survives. Session, Common, and Security middleware are irrelevant to the task and get dropped.

</td>
</tr>
</table>

```bash
$ cat django/middleware.py | squeez "Fix the CSRF validation bug in the referer check"
```

<details>
<summary><b>Another example — filtering git log</b></summary>

Task: *"Find the commit that changed the authentication timeout"*

**Before** — 25 commits of noise:
```
a1b2c3d Fix typo in README
e4f5g6h Update CI pipeline
i7j8k9l Bump version to 2.3.1
m0n1o2p Add docker-compose.yml
q3r4s5t Refactor database migrations
u6v7w8x Change auth timeout from 30m to 1h
y9z0a1b Fix linting warnings
c2d3e4f Update dependencies
...
```

**After** — the one commit that matters:
```
u6v7w8x Change auth timeout from 30m to 1h
```

```bash
$ git log --oneline -25 | squeez "find the commit that changed the authentication timeout"
```

</details>

## Installation

```bash
pip install squeez
```

For generative model training (Qwen + LoRA):

```bash
pip install -r requirements-train.txt
```

For encoder model training (mmBERT):

```bash
pip install -r requirements-encoder.txt
```

## Quick Start

### CLI

```bash
# Pipe tool output through squeez
cat output.txt | squeez "Fix the CSRF validation bug"

# Or with a file
squeez "Fix the CSRF bug" --input-file output.txt

# Explicit extract subcommand also works
squeez extract "Fix the CSRF bug" --input-file output.txt
```

### Python API

```python
from squeez.inference.extractor import ToolOutputExtractor

# Load model from config/env
extractor = ToolOutputExtractor()

# Or load a generative model locally
extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")

# Or load an encoder model (auto-detected from config.json)
extractor = ToolOutputExtractor(model_path="./output/squeez_encoder")

# Or connect to a server explicitly
extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1", model_name="squeez")

filtered = extractor.extract(
    task="Fix the CSRF validation bug in middleware",
    tool_output=raw_output,
)
print(filtered)  # Only the relevant lines
```

Both model types use the same `extract()` API. The generative model returns JSON (`{"relevant_lines": [...]}`), the encoder classifies each line directly. Both return filtered text.

### Configuration

Backend is resolved in order: CLI args > env vars > config file (`squeez.yaml` or `configs/default.yaml`).

```yaml
# squeez.yaml
backend: null  # auto-detect from model; or "transformers", "vllm", "encoder"
local_model_path: "./output/squeez_qwen"
# server_url: "https://api.groq.com/openai/v1"
# server_model: "squeez"
```

```bash
# Or via environment variables
export SQUEEZ_LOCAL_MODEL=./output/squeez_qwen
export SQUEEZ_SERVER_URL=https://api.groq.com/openai/v1
export SQUEEZ_SERVER_MODEL=squeez
export SQUEEZ_API_KEY=gsk_...
```

Clear flag names are available on the CLI, with the old names kept as aliases:

```bash
squeez "Fix the bug" --local-model ./output/squeez_qwen
squeez "Fix the bug" --server-url http://localhost:8000/v1 --server-model squeez
```

### Use with Claude Code

Add this to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global):

```
Always when you invoke a shell command, pipe it through `squeez` and tell exactly what you want to know.

Examples:
- `bun test 2>&1 | squeez "did the tests pass?"`
- `git log --oneline -50 | squeez "find the commit that broke CSRF"`
- `cat src/auth/middleware.py | squeez "find the referer validation logic"`

Do NOT use squeez when:
- You need exact, uncompressed output (e.g. writing a patch)
- The command is interactive
```

This saves context tokens by replacing verbose tool output with only the relevant lines.

Also works with other coding agents (Codex CLI, OpenCode, etc.) via their equivalent instruction files.

## Training

### 1. Download the dataset

```bash
python scripts/download_data.py
```

This pulls the [SWE-bench tool output dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) (7,148 train + 436 eval samples) from HuggingFace.

### 2a. Train generative model (Qwen + LoRA)

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/dev.jsonl
```

Default: Qwen 3.5 2B with LoRA (r=16, alpha=32). See `configs/default.yaml` for all hyperparameters.

### 2b. Train encoder model (mmBERT)

```bash
# Prepare encoder-format data from the ChatML training data
python scripts/prepare_encoder_data.py

# Train the encoder
python -m squeez.encoder.train \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model jhu-clsp/mmBERT-base \
    --output-dir output/squeez_encoder
```

The encoder is a 307M parameter mmBERT with a token classification head. It classifies each line as relevant/irrelevant and uses sliding windows to handle outputs longer than the 8K context.

### 3. Evaluate

```bash
# Generative model
squeez eval \
    --extractor-model output/squeez_qwen \
    --eval-file data/test.jsonl

# Encoder model
python -m squeez.encoder.evaluate \
    --model-path output/squeez_encoder \
    --eval-file data/encoder_test.jsonl
```

Both produce the same metrics format (span F1, ROUGE-L, compression ratio) for direct comparison.

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
squeez pipeline --phase 1 2 3 4 5 6 7 8 \
    --output-dir data \
    --github-token $GITHUB_TOKEN \
    --teacher-api-key $GROQ_API_KEY \
    --teacher-base-url https://api.groq.com/openai/v1
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
