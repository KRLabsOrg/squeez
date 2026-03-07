# Squeez

<p align="center">
  <img src="https://github.com/KRLabsOrg/squeez/blob/main/assets/squeez_mascot.png?raw=true" alt="Squeez Logo" width="300"/>
</p>

**Squeeze verbose LLM agent tool output down to only the relevant lines.**

LLM coding agents waste 80-95% of context tokens on irrelevant tool output. Squeez trains a small (2-3B) generative model to identify and extract only the lines that matter — compressing tool output by ~86% on average.

## Quick Example

```bash
$ cat django/middleware.py | squeez "Fix the CSRF validation bug"
class CsrfViewMiddleware(MiddlewareMixin):
    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
        if referer is None:
            raise RejectRequest('No referer')
        good_referer = request.get_host()
        if not same_origin(referer, good_referer):
            raise RejectRequest('Bad referer')
```

From 42 lines of middleware code, squeez extracts only the 8 lines relevant to the CSRF referer check.

## Highlights

- **~86% compression** — keeps only the lines the agent needs
- **CLI + Python API** — `cat file | squeez "task"` or `ToolOutputExtractor`
- **Two backends** — vLLM server (fast) or local transformers (no server)
- **Config-driven** — YAML config, env vars, or CLI args
- **Agent-ready** — works with Claude Code, Codex CLI, OpenCode via instruction files
- **Open dataset** — [7.5K samples](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) from real SWE-bench tool execution

## Links

- [GitHub](https://github.com/KRLabsOrg/squeez)
- [PyPI](https://pypi.org/project/squeez/)
- [HuggingFace Dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
