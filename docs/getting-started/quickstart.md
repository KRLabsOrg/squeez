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

# Or load model locally
extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")

# Or connect to a server explicitly
extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1", model_name="squeez")

filtered = extractor.extract(
    task="Fix the CSRF validation bug in middleware",
    tool_output=raw_output,
)
print(filtered)  # Only the relevant lines
```

## How it works

The model receives the task description and raw tool output, then returns a JSON object:

```json
{"relevant_lines": ["class CsrfViewMiddleware(MiddlewareMixin):", "    def _check_referer(self, request):", ...]}
```

The `extract()` method parses this JSON and joins the lines into filtered text.
