# Dataset

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## Data sources

The dataset combines:

1. **SWE real data** — Tool calls executed on cloned SWE-bench repositories. Real `git grep`, `pytest`, `pip install`, `mypy`, `git log`, etc. output. Labeled by a teacher LLM that writes a focused query and selects grounded spans over the original output.

2. **Synthetic multi-ecosystem** — LLM-generated raw tool output for ecosystems beyond Python: npm, TypeScript, Rust, Go, Java, Docker, Terraform, kubectl, and more. A teacher then produces focused queries plus grounded spans over that raw output.

## Sample format

The canonical source of truth is:

```json
{
  "instance_id": "django__django-11099",
  "source": "swe",
  "tool_type": "read_file",
  "query": "Find the referer validation block in the CSRF middleware.",
  "background_task": "Fix the CSRF validation bug...",
  "tool_output": "class CsrfViewMiddleware ...",
  "gold_spans": [
    {"start_line": 1, "end_line": 8, "reason": "relevant"}
  ],
  "is_irrelevant": false
}
```

From that canonical representation, Squeez derives two model-specific views.

For the main benchmark, positive rows are expected to have non-empty
`gold_spans`. If a positive sample cannot be answered from its tool output, it
is dropped after query fallback rather than preserved as an empty label.

### Qwen SFT view

`prompt` contains ChatML with a focused `query`, optional `background_task`, and raw `tool_output`:

```
<|im_start|>system
You prune verbose tool output for a coding agent...
<|im_end|>
<|im_start|>user
<query>
Find the referer validation block in the CSRF middleware.
</query>
<background_task>
Fix the CSRF validation bug in django...
</background_task>
<tool_output>
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        ...
</tool_output>
<|im_end|>
<|im_start|>assistant
```

`response` contains the extracted verbatim text wrapped in XML:

```xml
<relevant_lines>
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
</relevant_lines>
```

### Encoder view

```json
{
  "task": "Find the referer validation block in the CSRF middleware.",
  "tool_output": "class CsrfViewMiddleware ...",
  "relevant_lines": [
    "class CsrfViewMiddleware(MiddlewareMixin):",
    "    def _check_referer(self, request):"
  ],
  "tool_type": "read_file"
}
```

### Empty / irrelevant samples

Empty rows are reserved for explicit negatives, such as synthetic hard
negatives where the query is intentionally mismatched with the tool output.
Those rows store `gold_spans: []`, and the derived Qwen row becomes:

```xml
<relevant_lines>
</relevant_lines>
```

## Splits

SWE data is split by repository (zero repo overlap):

- **Test**: `pydata/xarray`, `pallets/flask`
- **Dev**: `psf/requests`
- **Train**: all others (django, sympy, scikit-learn, sphinx, matplotlib, pytest, astropy, pylint, seaborn)

Synthetic data is split per tool type: 10% test, 5% dev, 85% train. Hard negatives are capped in held-out splits so they do not dominate tool-level evaluation.

## Tool types

| Ecosystem | Tool types |
|-----------|-----------|
| **Python** | read_file, grep, python, test_output, pip_install, type_check, coverage, lint_output, build_output |
| **Git** | git_log, git_diff, git_blame, ls |
| **JavaScript/TypeScript** | npm_install, npm_build, tsc, eslint |
| **Rust** | cargo_build |
| **Go** | go_build |
| **Java** | mvn_gradle |
| **C/C++** | make_cmake |
| **Infrastructure** | docker_build, docker_logs, terraform, kubectl |
| **HTTP** | curl |
| **Python (synthetic)** | grep_output, git_log_output, git_diff_output, python_output, mypy_pyright |
