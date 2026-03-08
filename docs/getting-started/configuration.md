# Configuration

Squeez resolves backend configuration in order: **CLI args > env vars > config file**.

## Config file

Create a `squeez.yaml` in your project root, or use `configs/default.yaml`:

```yaml
backend: null  # auto-detect from model; or "transformers", "vllm", "encoder"
local_model_path: "./output/squeez_qwen"

# Or remote API backend
# server_url: "https://api.groq.com/openai/v1"
# server_model: "squeez"
```

Config files are searched in order:

1. `./squeez.yaml`
2. `./configs/default.yaml`
3. `~/.config/squeez/config.yaml`

## Environment variables

| Variable | Description |
|----------|-------------|
| `SQUEEZ_LOCAL_MODEL` | Path to local model directory |
| `SQUEEZ_SERVER_URL` | OpenAI-compatible API URL |
| `SQUEEZ_SERVER_MODEL` | Remote model ID on that server |
| `SQUEEZ_API_KEY` | API key (also checks `OPENAI_API_KEY`) |
| `SQUEEZ_BACKEND` | Optional backend preference: `transformers`, `vllm`, or `encoder` |

```bash
export SQUEEZ_LOCAL_MODEL=./output/squeez_qwen
# or
export SQUEEZ_SERVER_URL=https://api.groq.com/openai/v1
export SQUEEZ_SERVER_MODEL=squeez
export SQUEEZ_API_KEY=gsk_...
```

Legacy env vars `SQUEEZ_MODEL_PATH` and `SQUEEZ_BASE_URL` still work.

## CLI flags

CLI flags override everything:

```bash
squeez "Fix the bug" --local-model ./output/squeez_qwen --input-file output.txt
squeez "Fix the bug" --server-url http://localhost:8000/v1 --server-model squeez
```

Legacy flags `--model-path`, `--base-url`, and `--model-name` still work.

## Using with Groq

```bash
export SQUEEZ_SERVER_URL=https://api.groq.com/openai/v1
export SQUEEZ_API_KEY=gsk_...
cat output.txt | squeez "Fix the CSRF bug" --server-model openai/gpt-oss-120b
```
