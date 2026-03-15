# Squeez

<p align="center">
  <img src="https://github.com/KRLabsOrg/squeez/blob/main/assets/squeez_mascot.png?raw=true" alt="Squeez Logo" width="300"/>
  <br><em>Squeeze out the juice, leave the pulp behind.</em>
</p>

Squeeze verbose LLM agent tool output down to only the relevant evidence blocks.

[![PyPI](https://img.shields.io/pypi/v/squeez)](https://pypi.org/project/squeez/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow.svg)](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## The Problem

LLM coding agents waste **80-95% of context tokens** on irrelevant tool output. When an agent reads a 500-line file to find one function, or runs `git log` to find a specific commit, most of the output is noise.

Squeez trains small models to identify and extract only the lines that matter for the task at hand — compressing tool output by ~86% on average.

Three approaches are available:

- **Generative** (Qwen 3.5 2B + LoRA) — high-quality extraction via XML-wrapped verbatim output
- **Pooled encoder** (ModernBERT / ettin) — single-pass encoder with line-level mean-pool classification, works with any HuggingFace encoder
- **Token encoder** (mmBERT) — per-token binary classification with sliding window

## Example

Query: *"Find the referer validation block in the CSRF middleware"*

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
$ cat django/middleware.py | squeez "Find the referer validation block in the CSRF middleware"
```

<details>
<summary><b>Another example — filtering git log</b></summary>

Query: *"Find the commit that changed the authentication timeout"*

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

For encoder model training:

```bash
pip install -r requirements-encoder.txt
```

## Quick Start

### CLI

```bash
# Pipe tool output through squeez
cat output.txt | squeez "Find the failing traceback block"

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

# Or load an encoder model (pooled or token, auto-detected from config.json)
extractor = ToolOutputExtractor(model_path="./output/squeez_pooled")

# Or connect to a server explicitly
extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1", model_name="squeez")

filtered = extractor.extract(
    task="Find the referer validation block in middleware",
    tool_output=raw_output,
)
print(filtered)  # Only the relevant evidence block
```

Both model types use the same `extract()` API. Publicly the argument is still named `task`, but the intended input is a short focused extraction query or agent subgoal. The generative model returns XML-wrapped verbatim text internally, the encoder classifies lines directly. Both return filtered text.

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

This saves context tokens by replacing verbose tool output with only the relevant evidence block.

Also works with other coding agents (Codex CLI, OpenCode, etc.) via their equivalent instruction files.

## Training

### 1. Download the released dataset

```bash
python scripts/download_data.py
```

This pulls the released [tool output extraction dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) from HuggingFace.

### 2a. Train generative model (Qwen + LoRA)

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/dev.jsonl
```

Default: Qwen 3.5 2B with LoRA (r=16, alpha=32). See `configs/default.yaml` for all hyperparameters.

### 2b. Train pooled encoder (recommended)

```bash
python -m squeez.encoder.train \
    --classifier-type pooled \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-base \
    --output-dir output/squeez_pooled \
    --batch-size 96 \
    --gradient-accumulation-steps 2 \
    --max-length 4096 \
    --learning-rate 2e-5 \
    --num-epochs 4
```

The pooled encoder runs a single forward pass over the full input, mean-pools hidden states per line, and classifies each line as relevant/irrelevant. Works with any HuggingFace encoder model (ModernBERT, ettin, DeBERTa, etc.) and uses sliding windows for outputs longer than `--max-length`.

After training, the model can be loaded standalone without squeez installed:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("output/squeez_pooled", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("output/squeez_pooled")

result = model.process(
    task="Find the traceback that shows the import error",
    tool_output=open("output.log").read(),
    tokenizer=tokenizer,
)
print(result["highlighted_lines"])
```

### 2c. Train token encoder (alternative)

```bash
python -m squeez.encoder.train \
    --classifier-type token \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-base \
    --output-dir output/squeez_encoder
```

### 3. Evaluate

```bash
# Generative model (local)
squeez eval \
    --extractor-model output/squeez_qwen \
    --eval-file data/test.jsonl \
    --max-new-tokens 4096

# Generative model (remote vLLM server)
squeez eval \
    --server-url http://localhost:8000/v1 \
    --eval-file data/test.jsonl \
    --max-new-tokens 4096 \
    --request-concurrency 8

# Encoder model (pooled or token, auto-detected)
python -m squeez.encoder.evaluate \
    --model-path output/squeez_pooled \
    --eval-file data/encoder_test.jsonl \
    --examples-output eval_examples_pooled.json
```

All produce the same metrics format (strict and fuzzy line overlap, ROUGE-L, compression ratio) for direct comparison.

## Dataset

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

The current system uses one canonical source of truth:

- `query`: a short focused extraction request or agent subgoal
- `tool_output`: the raw tool output exactly as seen by the agent
- `gold_spans`: contiguous spans over the raw output

From that canonical format, Squeez derives:

- Qwen SFT files: `prompt + XML response`
- encoder files: `task/query + tool_output + relevant_lines`

This keeps training, evaluation, and QA grounded in verbatim source text. See the dataset card for the exact published split details after the next sync.

For the main benchmark, positive samples are expected to have non-empty
`gold_spans`. If a task-derived query does not yield extractable evidence,
Squeez retries with a tool-content-first query; if that still yields no spans,
the sample is dropped. Empty outputs are reserved for explicit negatives.

## Data Generation

The supported public path is fresh generation from scratch:

```bash
python scripts/build_full_dataset.py \
    --output-dir data/v3 \
    --teacher-model openai/gpt-oss-120b \
    --teacher-base-url http://localhost:8000/v1
```

This emits:

- `canonical_train/dev/test.jsonl`
- `train/dev/test.jsonl`
- `encoder_train/dev/test.jsonl`

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
