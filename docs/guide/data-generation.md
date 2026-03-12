# Data Generation

Squeez supports one public dataset-generation workflow:

- **fresh generation** from SWE-bench + synthetic generation

## Canonical format

The source of truth is:

- `query`
- `tool_output`
- `gold_spans`

Qwen and encoder datasets are derived from that canonical representation.

Positive samples should produce non-empty `gold_spans`. If a task-derived
query fails on a positive sample, the relabeler retries with a
tool-content-first query. If that still yields no spans, the sample is
dropped rather than kept as an empty label. Empty rows are reserved for
explicit negatives.

## Fresh build overview

```
SWE-bench instances
    │
    ▼
Phase 1: Load SWE-bench instances
    │
    ▼
Phase 2: Clone repos (bare git clones)
    │
    ▼
Phase 3: Generate tool calls (3-7 per instance)
    │
    ▼
Phase 4: Execute tool calls (real commands against repos)
    │
    ▼
Phase 5: Auto-label (heuristic relevance scoring)
    │
    ▼
Phase 6: LLM relabeling/distillation (teacher writes focused query + selects relevant spans)
    │
    ▼
Phase 7: Assemble canonical/Qwen/encoder splits
    │
    ▼
Phase 8: Validate dataset quality
```

## Build from scratch

```bash
python scripts/build_full_dataset.py \
    --output-dir data/v3 \
    --teacher-model openai/gpt-oss-120b \
    --teacher-base-url http://localhost:8000/v1
```

## Key design decisions

### Real tool execution
SWE tool calls are executed as real commands (`git grep`, `git blame`, `git log`, `pytest`, `ruff`, `python`, etc.) against bare-cloned repos checked out at the correct SWE-bench base commit. Synthetic samples produce realistic raw tool output, but the final labels are still grounded as spans over that raw output.

### Grounded labels
The teacher model writes a focused extraction query and returns contiguous spans over a numbered view of the output. Those spans are mapped back onto the original raw output. Canonical labels never store synthetic line numbers.

### Content-first fallback
The preferred query is still derived from the original task, but only when the
tool output can actually answer it. If the task-derived query yields no spans
for a positive sample, Squeez retries with a query driven primarily by the
tool content itself. This is especially important for reused raw outputs where
the original tool/file selection may have been noisy.

### One source of truth
Canonical rows are converted into:

- Qwen SFT rows (`prompt`, XML `response`)
- encoder rows (`task`, `tool_output`, `relevant_lines`)

so training, evaluation, and QA all derive from the same grounded spans.

## Tool types

| Tool Type | Description |
|-----------|--------|-------------|
| read_file | Source file contents via `git show` |
| grep | Code search via `git grep` |
| python | Python command execution |
| git_log | Commit history |
| test_output | Test runner output |
| git_diff | Diff output |
| git_blame | Line-level attribution |
| ls | Directory listings |
| lint_output | Linter warnings (ruff) |
| build_output | Build/compile output |
| curl | HTTP responses |
