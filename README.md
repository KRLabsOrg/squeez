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

Squeez trains a small (2-3B) generative model to identify and extract only the lines that matter for the task at hand — compressing tool output by ~86% on average.

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

Backend is resolved in order: CLI args > env vars > config file (`squeez.yaml` or `configs/default.yaml`).

```yaml
# squeez.yaml
model_path: "./output/squeez_qwen"     # local transformers
# base_url: "https://api.groq.com/openai/v1"  # or remote API
```

```bash
# Or via environment variables
export SQUEEZ_MODEL_PATH=./output/squeez_qwen
export SQUEEZ_BASE_URL=https://api.groq.com/openai/v1
export SQUEEZ_API_KEY=gsk_...
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
