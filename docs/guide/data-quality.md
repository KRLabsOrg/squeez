# Data Quality & Review Process

The dataset powering Squeez now uses canonical `query + tool_output + gold_spans` rows as the source of truth. This page documents the quality checks that matter for that format.

## Pipeline Overview

```
Raw tool output
    │
    ▼
Teacher relabeling (focused query + grounded spans)
    │
    ▼
Canonical rows (`query`, `tool_output`, `gold_spans`)
    │
    ▼
Derived Qwen / encoder views
    │
    ▼
Automated QA + manual review
```

## Automated QA

Every canonical sample should pass:

- **Grounding**: every span lands inside the raw `tool_output`
- **Ordering**: spans are sorted and non-overlapping
- **Canonical derivation**: Qwen XML and encoder `relevant_lines` are both derived from the same spans
- **No synthetic numbering leakage**: final labels must match raw output, not temporary numbered views
- **Metadata consistency**: counts and compression are recomputed from the derived content

Earlier dataset versions mixed canonical truth with model-specific wrappers and temporary numbering views. The current system is designed specifically to avoid that class of bug by keeping `gold_spans` canonical and deriving all other representations from them.

## Review priorities

When reviewing held-out samples, the main questions are:

1. Is the `query` a realistic next-step extraction request?
2. Do the `gold_spans` point to the smallest useful evidence block?
3. Would an agent plausibly want this exact block next?
4. If the sample is empty/irrelevant, is it an explicit negative rather than a failed positive?

## Recommendations for Contributors

If you want to further improve the dataset:

1. **Keep canonical rows clean**: fix `query`/`gold_spans`, not only derived XML.
2. **Check grounding first**: labels must map back to raw output exactly.
3. **Review negatives carefully**: empty labels should be deliberate explicit negatives, not artifacts of bad task generation.
4. **Prefer focused extraction subgoals** over full issue statements.
5. **Use content-first fallback for positives**: if a task-derived query is not answerable from a tool output, retry with a tool-aware query driven by the output itself; if that still fails, drop the sample.
