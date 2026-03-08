# Quick Start

## CLI

Pipe any tool output through squeez with a task description:

```bash
# Filter a source file
cat src/auth/middleware.py | squeez "Fix the CSRF validation bug"

# Filter git log
git log --oneline -50 | squeez "Find the commit that broke auth"

# Filter test output
pytest tests/ 2>&1 | squeez "Why is test_login failing"

# From a file
squeez "Fix the bug" --input-file output.txt

# Explicit extract subcommand also works
squeez extract "Fix the bug" --input-file output.txt
```

## Python API

```python
from squeez.inference.extractor import ToolOutputExtractor

# Load backend from config or env
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

## How it works

Squeez supports two model types behind the same `extract()` API:

- **Generative** (default): The model returns a JSON object `{"relevant_lines": [...]}` and `extract()` joins them into text.
- **Encoder**: A token classifier labels each line as relevant/irrelevant. Uses sliding windows for outputs longer than the context window.

The backend is auto-detected from the model's `config.json`. You can also set it explicitly via `SQUEEZ_BACKEND=encoder` or `backend: "encoder"` in config.
