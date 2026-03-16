# Squeez

<p align="center">
  <img src="https://github.com/KRLabsOrg/squeez/blob/main/assets/squeez_mascot.png?raw=true" alt="Squeez Logo" width="300"/>
  <br><em>Squeeze out the juice, leave the pulp behind.</em>
</p>

LLM coding agents waste 80-95% of context tokens on irrelevant tool output. Squeez extracts only the lines that matter, compressing tool output by ~91% while keeping 86% of the relevant information.

[![PyPI](https://img.shields.io/pypi/v/squeez)](https://pypi.org/project/squeez/)
[![Model](https://img.shields.io/badge/HF-Squeez--2B-yellow.svg)](https://huggingface.co/KRLabsOrg/squeez-2b)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow.svg)](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## How it works

Squeez uses a fine-tuned Qwen 3.5 2B model to read tool output alongside a task description and return only the relevant lines.

### Example: filtering test output

Task: *"Find the test failure related to authentication"*

<table>
<tr>
<th>Before (45 lines, ~1,500 tokens)</th>
<th>After (6 lines, ~200 tokens)</th>
</tr>
<tr>
<td>

```
$ python -m pytest tests/ -v
======================== test session starts ========================
platform linux -- Python 3.12.1, pytest-8.1.1
collected 23 items

tests/test_auth.py::test_login_valid PASSED
tests/test_auth.py::test_login_invalid PASSED
tests/test_auth.py::test_token_refresh FAILED
tests/test_auth.py::test_logout PASSED
tests/test_users.py::test_create_user PASSED
tests/test_users.py::test_delete_user PASSED
tests/test_users.py::test_list_users PASSED
tests/test_middleware.py::test_csrf_check PASSED
tests/test_middleware.py::test_rate_limit PASSED
tests/test_middleware.py::test_cors_headers PASSED

======================= FAILURES ================================
_____ test_token_refresh ________________________________________

    def test_token_refresh(self):
        token = self.client.get_token(expired=True)
>       refreshed = self.client.refresh(token)
E       AuthenticationError: Token refresh window expired
E       Expected: new token within 30m window
E       Got: rejection after 15m (timeout changed?)

tests/test_auth.py:47: AuthenticationError
================ short test summary info ========================
FAILED tests/test_auth.py::test_token_refresh
================== 1 failed, 9 passed ==========================
```

</td>
<td>

```
tests/test_auth.py::test_token_refresh FAILED

    def test_token_refresh(self):
        token = self.client.get_token(expired=True)
>       refreshed = self.client.refresh(token)
E       AuthenticationError: Token refresh window expired
E       Expected: new token within 30m window
E       Got: rejection after 15m (timeout changed?)
```

**87% compression.** Only the failing test and its traceback survive.

</td>
</tr>
</table>

```bash
$ python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

<details>
<summary><b>More examples</b></summary>

**Filtering git log:**

```
$ git log --oneline -25 | squeez "find the commit that changed the authentication timeout"

u6v7w8x Change auth timeout from 30m to 1h
```

**Filtering build output:**

```
$ npm run build 2>&1 | squeez "find the TypeScript error"

src/components/Auth.tsx(34,5): error TS2345: Argument of type 'string' is
  not assignable to parameter of type 'AuthToken'.
```

**Filtering kubectl output:**

```
$ kubectl describe pod api-server-7d4b | squeez "why is the pod failing"

    State:          Waiting
      Reason:       CrashLoopBackOff
    Last State:     Terminated
      Reason:       Error
      Exit Code:    1
  Warning  BackOff  3m (x5)  kubelet  Back-off restarting failed container
```

</details>

## Results

Evaluated on 617 held-out test samples from SWE-bench, across 14 tool types:

| Model | Precision | Recall | F1 | Compression |
|-------|-----------|--------|------|-------------|
| **Squeez-2B** | **0.8043** | **0.8624** | **0.7895** | 0.9150 |
| Qwen 3.5 35B A3B (zero-shot) | 0.7402 | 0.7498 | 0.7000 | 0.9177 |
| Qwen 3.5 2B (untrained) | 0.4154 | 0.5299 | 0.4075 | 0.8197 |
| BM25 (10%) | 0.1277 | 0.2172 | 0.1314 | 0.9036 |
| Random (10%) | 0.0738 | 0.1009 | 0.0697 | 0.9067 |

Squeez-2B (2B params) outperforms a 35B MoE model at zero-shot and is 6x better than BM25 on Span F1.

## Install

```bash
pip install squeez
```

## Quick start

### With vLLM (recommended)

```bash
# Start the server
pip install vllm
vllm serve KRLabsOrg/squeez-2b --dtype bfloat16 --max-model-len 16384

# Use from squeez CLI
pip install squeez
export SQUEEZ_SERVER_URL=http://localhost:8000/v1
cat output.txt | squeez "find the bug"

# Or pipe directly
python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

vLLM keeps the model warm in memory with batched inference and high throughput.

### Local inference (no server)

```bash
pip install squeez

cat output.txt | squeez "Find the failing traceback block"
squeez "Fix the CSRF bug" --input-file output.txt
```

> **Note:** Local mode loads the model on every call. Fine for one-off use, but for repeated calls (e.g. an agent piping every tool through squeez), use vLLM.

### Any OpenAI-compatible API

Works with Groq, Together, or any OpenAI-compatible server. Set the URL, model name, and API key:

```bash
export SQUEEZ_SERVER_URL=https://api.groq.com/openai/v1
export SQUEEZ_SERVER_MODEL=squeez
export SQUEEZ_API_KEY=gsk_...
```

### Python API

```python
from squeez.inference.extractor import ToolOutputExtractor

# Default: loads KRLabsOrg/squeez-2b locally
extractor = ToolOutputExtractor()

# Or connect to a server
extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1")

# Or use a custom local model
extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")

filtered = extractor.extract(
    task="Find the referer validation block",
    tool_output=raw_output,
)
```

### Use with Claude Code

Add to your `CLAUDE.md`:

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

Works with other coding agents (Codex CLI, OpenCode, etc.) via their equivalent instruction files.

---

## Advanced

<details>
<summary><b>Configuration</b></summary>

Resolved in order: CLI flags > environment variables > config file.

Config file is loaded from the first found: `./squeez.yaml`, `./configs/default.yaml`, `~/.config/squeez/config.yaml`.

```yaml
# squeez.yaml
server_url: "http://localhost:8000/v1"
# local_model_path: "./output/squeez_qwen"  # for local inference instead
# backend: null  # auto-detect; or "transformers", "vllm", "encoder"
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `SQUEEZ_SERVER_URL` | Server URL (vLLM, Ollama, etc.) |
| `SQUEEZ_LOCAL_MODEL` | Path to local model directory |
| `SQUEEZ_SERVER_MODEL` | Model name on the server |
| `SQUEEZ_API_KEY` | API key (if needed) |
| `SQUEEZ_BACKEND` | Force backend: `transformers`, `vllm`, `encoder` |

</details>

<details>
<summary><b>Encoder models</b></summary>

Squeez also supports encoder-based extraction (ModernBERT, etc.) as an alternative to the generative model. These are faster but less accurate.

Two encoder approaches:
- **Token encoder**: per-token binary classification, aggregated per line via max-pool
- **Pooled encoder**: single-pass encoder with line-level mean-pool classification

```python
from squeez.inference.extractor import ToolOutputExtractor

extractor = ToolOutputExtractor(model_path="./output/squeez_encoder")
filtered = extractor.extract(task="Find the bug", tool_output=raw_output)
```

Standalone loading without squeez installed:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("output/squeez_pooled", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("output/squeez_pooled")

result = model.process(
    task="Find the traceback",
    tool_output=open("output.log").read(),
    tokenizer=tokenizer,
)
print(result["highlighted_lines"])
```

</details>

<details>
<summary><b>Training</b></summary>

See [TRAINING.md](TRAINING.md) for full training and evaluation commands.

```bash
# Download dataset
python scripts/download_data.py

# Train generative model (Qwen 3.5 2B + LoRA)
squeez train --train-file data/train.jsonl --eval-file data/dev.jsonl

# Train token encoder
python -m squeez.encoder.train \
    --classifier-type token \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-base \
    --output-dir output/squeez_encoder

# Evaluate
squeez eval --extractor-model output/squeez_qwen --eval-file data/test.jsonl
```

</details>

<details>
<summary><b>Dataset</b></summary>

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

Built from SWE-bench repositories. Each sample has:
- `query`: a focused extraction request or agent subgoal
- `tool_output`: raw tool output as seen by the agent
- `gold_spans`: contiguous spans over the raw output

From this canonical format, Squeez derives generative SFT files and encoder training files.

To regenerate from scratch:

```bash
python scripts/build_full_dataset.py \
    --output-dir data/v3 \
    --teacher-model openai/gpt-oss-120b \
    --teacher-base-url http://localhost:8000/v1
```

</details>

## Citation

```bibtex
@software{kovacs2026squeez,
    title={Squeez: Compressing Tool Output for LLM Coding Agents},
    author={Adam Kovacs},
    year={2026},
    url={https://github.com/KRLabsOrg/squeez}
}
```

## License

Apache 2.0
