# Squeez

<p align="center">
  <img src="https://github.com/KRLabsOrg/squeez/blob/main/assets/squeez_mascot.png?raw=true" alt="Squeez Logo" width="300"/>
  <br><em>Squeeze out the juice, leave the pulp behind.</em>
</p>

[![PyPI](https://img.shields.io/pypi/v/squeez)](https://pypi.org/project/squeez/)
[![Model](https://img.shields.io/badge/HF-Squeez--2B-yellow.svg)](https://huggingface.co/KRLabsOrg/squeez-2b)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow.svg)](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

- Tool output pruner for LLM coding agents
- Pipe any tool output (pytest, grep, git log, npm build, kubectl, ...) through squeez with a task description, get back only the relevant lines
- Two models, same CLI: a **generative** Qwen 3.5 2B (0.80 F1, 92% compression) or a smaller **extractive** ModernBERT alternative
- CLI pipe, Python library, or vLLM server

Existing context pruning tools ([SWE-Pruner](https://github.com/Ayanami1314/swe-pruner), [Zilliz Semantic Highlight](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1), [Provence](https://arxiv.org/abs/2501.16214)) are built for source code or document paragraphs. They don't handle the mixed, unstructured format of tool output (stack traces interleaved with passing tests, grep matches with context lines, build logs with timestamps). Squeez is trained on 27 types of tool output from real SWE-bench workflows and synthetic multi-ecosystem observations.

```bash
pip install squeez
python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

## Example

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

Evaluated on 618 manually curated held-out examples spanning 27 tool types:

| Model | Precision | Recall | F1 | Compression |
|-------|-----------|--------|------|-------------|
| **Squeez-2B** | **0.80** | **0.86** | **0.80** | 0.92 |
| Qwen 3.5 35B A3B (zero-shot) | 0.74 | 0.75 | 0.73 | 0.92 |
| Kimi K2 (zero-shot) | 0.61 | 0.53 | 0.68 | 0.94 |
| Qwen 3.5 2B (untrained) | 0.42 | 0.53 | 0.55 | 0.82 |
| BM25 (10%) | 0.13 | 0.22 | 0.23 | 0.90 |
| Random (10%) | 0.07 | 0.10 | 0.20 | 0.91 |

Squeez-2B (2B params) outperforms the 18x larger Qwen 3.5 35B A3B by 11 recall points at the same compression level.

## Quick start

### With vLLM (recommended)

```bash
pip install vllm
vllm serve KRLabsOrg/squeez-2b --dtype bfloat16 --max-model-len 16384

# Use from squeez CLI
pip install squeez
export SQUEEZ_SERVER_URL=http://localhost:8000/v1
cat output.txt | squeez "find the bug"
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
| `SQUEEZ_BACKEND` | Force backend (rarely needed; auto-detected from the model) |

</details>

<details>
<summary><b>Use the extractive model instead</b></summary>

If you don't need the 2B generative model, point squeez at a smaller
extractive one — same CLI, same Python API. Configure once, then use
`squeez` normally:

```bash
export SQUEEZ_LOCAL_MODEL=KRLabsOrg/verbatim-rag-modern-bert-v2

pytest -q 2>&1 | squeez "find the failing test"
git log --oneline -50 | squeez "find the auth commit"
```

`KRLabsOrg/verbatim-rag-modern-bert-v2` is a 150M ModernBERT span model
trained on a multi-domain mix that includes Squeez tool-output. See
[RESULTS.md](RESULTS.md) for the head-to-head with Squeez-2B.

To train your own extractive model, see [TRAINING.md](TRAINING.md).

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

Built from SWE-bench repositories and synthetic multi-ecosystem tool outputs. Each sample has:
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
@misc{kovács2026squeeztaskconditionedtooloutputpruning,
      title={Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents}, 
      author={Ádám Kovács},
      year={2026},
      eprint={2604.04979},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2604.04979}, 
}
```

## License

Apache 2.0
