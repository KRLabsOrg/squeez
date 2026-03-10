# Dataset

Training data: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)

## Statistics

| | Count |
|---|---|
| **Train samples** | 8,241 |
| **Dev samples** | 252 |
| **Test samples** | 557 |
| **Total** | 9,050 |
| **SWE-bench real data** | 5,936 |
| **Synthetic multi-ecosystem** | 2,039 |
| **Synthetic SWE-style** | 1,075 |
| **Tool types** | 30 |

## Data sources

The dataset combines three sources:

1. **SWE-bench real data** — Tool calls executed on 2,294 cloned Python repos from SWE-bench (django, scikit-learn, sympy, etc.). Real `git grep`, `pytest`, `pip install`, `mypy` output. Labeled by a teacher LLM that selects relevant line spans grounded in the original output.

2. **Synthetic multi-ecosystem** — LLM-generated tool output for ecosystems beyond Python: npm, TypeScript, Rust, Go, Java, Docker, Terraform, kubectl, and more. Two-pass generation: Pass 1 generates task + output, Pass 2 picks relevant lines.

3. **Synthetic SWE-style** — LLM-generated versions of Python tool types that had poor quality in the real data (high noise rates from environment failures).

## Sample format

Each sample has three fields:

### prompt

System prompt + task description + tool output, formatted with Qwen ChatML tokens:

```
<|im_start|>system
You extract relevant lines from tool output for a coding task. Return the relevant lines inside <relevant_lines> tags.
<|im_end|>
<|im_start|>user
<task>
Fix the CSRF validation bug in django...
</task>
<tool_output>
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        ...
</tool_output>
<|im_end|>
<|im_start|>assistant
```

### response

Relevant lines wrapped in XML tags:

```xml
<relevant_lines>
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
</relevant_lines>
```

Or when the output is not relevant to the task:

```xml
<relevant_lines>
</relevant_lines>
```

### metadata

```json
{
    "instance_id": "django__django-11099",
    "tool_type": "read_file",
    "source": "swe",
    "num_total_lines": 42,
    "num_relevant_lines": 8,
    "compression_ratio": 0.81
}
```

The `source` field is one of `swe`, `synthetic`, or `synthetic_negative` (hard negatives where the task is intentionally mismatched with the tool output).

## Splits

**SWE-bench data** split by repository (zero instance overlap):

- **Test**: `pydata/xarray`, `pallets/flask`
- **Dev**: `psf/requests`
- **Train**: all others (django, sympy, scikit-learn, sphinx, matplotlib, pytest, astropy, pylint, seaborn)

**Synthetic data** split per tool type: 10% test, 5% dev, 85% train. Hard negatives capped at ~10% per tool type in test.

The held-out test set was manually curated: 61 overly broad annotations were excluded.

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
