# Dataset

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## Statistics

| | Count |
|---|---|
| **Train samples** | 7,148 |
| **Eval samples** | 436 |
| **Unique SWE-bench instances** | 2,282 |
| **With relevant lines** | 3,985 (53%) |
| **Empty (not relevant)** | 3,599 (47%) |
| **Avg relevant lines** | 34.2 (when non-empty) |
| **Avg compression** | 86% |

## Sample format

Each sample has three fields:

### prompt

System prompt + task description + raw tool output, formatted with chat template tokens:

```
<|system|>
You extract relevant lines from tool output for a coding task. Return a JSON object: {"relevant_lines": ["line1", "line2", ...]}. Include ONLY lines the agent needs to see.
<|user|>
Task: Fix the CSRF validation bug in django...

class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        ...
<|assistant|>
```

### response

JSON with the relevant lines:

```json
{"relevant_lines": ["class CsrfViewMiddleware(MiddlewareMixin):", "    def _check_referer(self, request):", ...]}
```

Or when the output is not relevant to the task:

```json
{"relevant_lines": []}
```

### metadata

```json
{
    "instance_id": "django__django-11099",
    "tool_type": "read_file",
    "compression_ratio": 0.87,
    "num_total_lines": 42,
    "num_relevant_lines": 8
}
```

## Splits

- **Train**: 10 repos (django, scikit-learn, sphinx, matplotlib, pytest, astropy, pylint, requests, seaborn, sympy)
- **Eval**: 2 held-out repos (xarray, flask) — zero repo overlap with train
