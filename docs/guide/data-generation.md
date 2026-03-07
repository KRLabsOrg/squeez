# Data Generation

The training dataset is built from real tool execution against SWE-bench repositories.

## Pipeline overview

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
Phase 6: LLM distillation (teacher selects relevant spans)
    │
    ▼
Phase 7: Assemble into train/eval JSONL
    │
    ▼
Phase 8: Validate dataset quality
```

## Running the full pipeline

```bash
squeez pipeline --phase 1 2 3 4 5 6 7 8 \
    --output-dir data \
    --github-token $GITHUB_TOKEN \
    --teacher-api-key $GROQ_API_KEY \
    --teacher-base-url https://api.groq.com/openai/v1
```

## Running individual phases

```bash
# Just phase 4 (execute tool calls)
squeez pipeline --phase 4 --output-dir data

# Phase 6 (LLM distillation) with custom concurrency
squeez pipeline --phase 6 --output-dir data --concurrency 5
```

## Key design decisions

### Real tool execution
All tool calls are executed as real commands (`git grep`, `git blame`, `git log`, `pytest`, `ruff`, `python`, etc.) against bare-cloned repos checked out at the correct SWE-bench base commit. No simulated output.

### Zero-hallucination extraction
The teacher model (gpt-oss-120b) returns JSON spans (`{"spans": [{"start": N, "end": M}]}`), which are matched against the original output to extract actual text lines. The student never sees generated text — only real lines from the original output.

### Repo-based train/eval split
Train and eval splits have zero repo overlap. Eval repos (xarray, flask) are entirely held out.

## Tool types

| Tool Type | Weight | Description |
|-----------|--------|-------------|
| read_file | 25% | Source file contents via `git show` |
| grep | 15% | Code search via `git grep` |
| python | 10% | Python command execution |
| git_log | 10% | Commit history |
| test_output | 10% | Test runner output |
| git_diff | 5% | Diff output |
| git_blame | 5% | Line-level attribution |
| ls | 5% | Directory listings |
| lint_output | 5% | Linter warnings (ruff) |
| build_output | 5% | Build/compile output |
| curl | 5% | HTTP responses |
