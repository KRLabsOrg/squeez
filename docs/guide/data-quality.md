# Data Quality & Review Process

The dataset powering squeez goes through a multi-stage quality assurance pipeline. This page documents the process, the issues found, and the corrections applied.

## Pipeline Overview

```
Raw SWE-bench tool output
    │
    ▼
Teacher distillation (gpt-oss-120b selects spans)
    │
    ▼
Zero-hallucination extraction (spans → actual text lines)
    │
    ▼
Automated QA (structural validation)
    │
    ▼
Manual review (test split only)
    │
    ▼
Final dataset on HuggingFace
```

## Automated QA

Every sample passes structural checks:

- **JSON validity**: response parses as `{"relevant_lines": [...]}`
- **Grounding**: every line in `relevant_lines` exists in the prompt (the raw tool output)
- **Ordering**: selected lines appear in the same order as in the original output
- **Metadata consistency**: `num_relevant_lines` matches the actual count in the response
- **Deduplication**: no duplicate lines in `relevant_lines`

### Metadata Fix (v0.1.1)

The initial release had stale metadata inherited from the distillation phase. Specifically:

- `num_relevant_lines` was computed before post-processing (filtering `...` separators and empty lines)
- `num_total_lines` came from upstream span counts, not the actual tool output
- `compression_ratio` was derived from the stale counts

**Impact**: 3,755 / 7,148 train samples and 230 / 436 test samples had incorrect `num_relevant_lines`. The actual `prompt` and `response` content (what the model trains on) was always correct — only the metadata was wrong.

**Fix**: All three metadata fields are now recomputed from the actual assembled content. The assembler (`sample_assembler.py`) was also patched to prevent this in future regenerations.

## Manual Review (Test Split)

The full 436-sample test split was manually reviewed in 5 batches. Each sample was checked against:

1. The task statement (SWE-bench issue)
2. The raw tool output embedded in the prompt
3. The selected `relevant_lines`

Corrections were only applied when the selected span was **clearly wrong** or **clearly too incomplete to be useful**.

### Results

| Metric | Count |
|---|---|
| Total samples reviewed | 436 / 436 |
| Corrected | 55 (12.6%) |
| Unchanged | 381 (87.4%) |

### Batch Breakdown

| Batch | Range | Reviewed | Corrected | Unchanged |
|---|---|---|---|---|
| 1 | 0–99 | 100 | 12 | 88 |
| 2 | 100–199 | 100 | 11 | 89 |
| 3 | 200–299 | 100 | 20 | 80 |
| 4 | 300–399 | 100 | 8 | 92 |
| 5 | 400–435 | 36 | 4 | 32 |

### Main Error Classes

The dominant issue was **not** random annotation noise. It was a repeated failure mode from the teacher distillation:

1. **Empty `python` labels despite clear tracebacks** — the teacher returned no spans when the output was a full traceback, even though the traceback is the most relevant signal for debugging
2. **Overly truncated traceback labels** — only the tail of a traceback was selected, missing the root cause
3. **Missed blocking failures** — labels selected unrelated context instead of the actual error
4. **Short error spans** — `test_output` samples where the `ImportError` or `E ...` failure lines were omitted

Most corrections involved Python tracebacks from environment/runtime failures (import-time errors that prevented the issue-specific reproduction from running).

### QA Metadata

Every test sample now carries per-sample QA metadata in `metadata.qa`:

```json
{
  "status": "pass",
  "response_json_valid": true,
  "relevant_lines_present_in_prompt": true,
  "relevant_lines_in_prompt_order": true,
  "duplicate_relevant_line_count": 0,
  "manual_review_batch1": {
    "batch": 1,
    "status": "reviewed_no_change",
    "rationale": "..."
  }
}
```

Corrected samples include the review status and rationale explaining what was changed and why.

## Training Split — Traceback Curation

After the test split review revealed a pattern of missing/truncated traceback labels, a targeted curation pass was applied to the training split. This was **not** a full manual review — it specifically fixed the highest-confidence traceback-related label errors.

### Scope

Only three categories were touched:

| Category | Before | After | Corrected |
|---|---|---|---|
| Empty `python` labels with clear traceback/error block | 64 | 0 | 64 |
| Short `python` spans with clear traceback | 52 | 0 | 52 |
| Short `test_output` spans with clear error block | 7 | 0 | 7 |
| **Total** | | | **123** |

Out of 6,790 train samples, 123 (1.8%) were corrected. The remaining 6,667 samples were reviewed by the heuristic and left unchanged.

### What Was Fixed

Corrected samples now properly include the traceback or error block that was previously omitted or truncated. These were dominated by:

- Missing compiled extensions (`ImportError` from C extensions)
- Missing optional dependencies
- Import failures from Python/package version incompatibilities
- pytest configuration/import failures with clear traceback output

### What Was Not Fixed

This pass deliberately did **not** touch:

- Subtle semantic selection mistakes in non-traceback samples
- Ambiguous cases where multiple spans could plausibly be selected
- Empty labels for `read_file`, `grep`, or `git_log` tools, where emptiness may be correct
- General semantic relabeling across all tool types

### QA Metadata

Every train sample now carries curation metadata in `metadata.qa.traceback_train_curation_v1`:

```json
{
  "status": "corrected_traceback_candidate"
}
```

or:

```json
{
  "status": "reviewed_no_change"
}
```

### Artifacts

- Curation script: `scripts/curate_train_tracebacks.py`
- Curation notes: `data/train_curated_tracebacks_v1_notes.json`
- Full report: `data/train_curated_tracebacks_v1_report.md`

## Summary of All Data Quality Passes

| Split | Pass | Samples | Corrected | Method |
|---|---|---|---|---|
| Both | Metadata fix | 7,584 | 3,985 metadata fields | Automated recomputation |
| Test | Manual review | 436 | 55 (12.6%) | Human review, 5 batches |
| Train | Traceback curation | 6,790 | 123 (1.8%) | Heuristic + targeted fix |

## Recommendations for Contributors

If you want to further improve the dataset:

1. **Structural fixes**: run the automated QA and fix any structural issues
2. **Candidate mining**: use heuristics to find high-confidence correction candidates
3. **Manual review**: review candidates and apply corrections where the label is clearly wrong
4. **Do not** blindly auto-relabel the entire training set — the judgment calls on what constitutes "relevant" output require context about the task
