"""Evaluate baselines on the squeez test set.

Runs naive baselines (random, first-N, BM25) and off-the-shelf models
(SWE-Pruner, Zilliz semantic-highlight, GLiNER2) against the same test
set and metrics used for squeez evaluation.

Usage:
    # Run only naive baselines (no GPU needed)
    python scripts/evaluate_baselines.py \
        --eval-file data/v3/encoder_test.jsonl \
        --baselines naive

    # Run SWE-Pruner (needs: pip install swe-pruner, CUDA + flash-attn)
    python scripts/evaluate_baselines.py \
        --eval-file data/v3/encoder_test.jsonl \
        --baselines swe_pruner

    # Run Zilliz semantic-highlight
    python scripts/evaluate_baselines.py \
        --eval-file data/v3/encoder_test.jsonl \
        --baselines zilliz

    # Run GLiNER2 (needs: pip install gliner2)
    python scripts/evaluate_baselines.py \
        --eval-file data/v3/encoder_test.jsonl \
        --baselines gliner2

    # Run all
    python scripts/evaluate_baselines.py \
        --eval-file data/v3/encoder_test.jsonl \
        --baselines all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics

logger = logging.getLogger(__name__)

ALL_NAIVE = ["random", "first_n", "last_n", "bm25"]
ALL_MODEL = ["swe_pruner", "zilliz", "gliner2"]
ALL_BASELINES = ALL_NAIVE + ALL_MODEL


# ---------------------------------------------------------------------------
# Metrics (reuse squeez internals)
# ---------------------------------------------------------------------------


def _compute_metrics(pred_lines: list[str], ref_lines: list[str], tool_output: str) -> dict:
    from squeez.training.evaluate import (
        compute_compression_ratio,
        compute_empty_accuracy,
        compute_fuzzy_span_metrics,
        compute_partial_overlap,
        compute_rouge_l,
        compute_span_metrics,
    )

    span = compute_span_metrics(pred_lines, ref_lines)
    fuzzy = compute_fuzzy_span_metrics(pred_lines, ref_lines, threshold=0.5)
    partial = compute_partial_overlap(pred_lines, ref_lines)
    empty = compute_empty_accuracy(pred_lines, ref_lines)
    pred_text = "\n".join(pred_lines)
    ref_text = "\n".join(ref_lines)
    rouge = compute_rouge_l(pred_text, ref_text)
    compression = compute_compression_ratio(tool_output, pred_text)

    return {
        "span_precision": span["precision"],
        "span_recall": span["recall"],
        "span_f1": span["f1"],
        "exact_match": span["exact_match"],
        "fuzzy_span_precision": fuzzy["precision"],
        "fuzzy_span_recall": fuzzy["recall"],
        "fuzzy_span_f1": fuzzy["f1"],
        "partial_overlap": partial,
        "empty_accuracy": empty["correct"],
        "empty_category": empty["category"],
        "rouge_l": rouge,
        "compression": compression,
    }


def _aggregate(all_metrics: list[dict]) -> dict:
    keys = [
        "span_precision",
        "span_recall",
        "span_f1",
        "exact_match",
        "fuzzy_span_precision",
        "fuzzy_span_recall",
        "fuzzy_span_f1",
        "partial_overlap",
        "empty_accuracy",
        "rouge_l",
        "compression",
    ]
    result = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        result[key] = {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
        }

    empty_confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for m in all_metrics:
        empty_confusion[m["empty_category"]] += 1
    result["empty_confusion"] = empty_confusion
    result["num_samples"] = len(all_metrics)
    return result


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------


def baseline_random(task: str, tool_output: str, target_ratio: float = 0.10) -> list[str]:
    """Keep ~10% of lines at random (matching typical gold ratio)."""
    lines = tool_output.split("\n")
    n_keep = max(1, int(len(lines) * target_ratio))
    if len(lines) <= n_keep:
        return lines
    return [lines[i] for i in sorted(random.sample(range(len(lines)), n_keep))]


def baseline_first_n(task: str, tool_output: str, target_ratio: float = 0.10) -> list[str]:
    """Keep the first ~10% of lines (naive truncation)."""
    lines = tool_output.split("\n")
    n_keep = max(1, int(len(lines) * target_ratio))
    return lines[:n_keep]


def baseline_last_n(task: str, tool_output: str, target_ratio: float = 0.10) -> list[str]:
    """Keep the last ~10% of lines."""
    lines = tool_output.split("\n")
    n_keep = max(1, int(len(lines) * target_ratio))
    return lines[-n_keep:]


def baseline_bm25(task: str, tool_output: str, target_ratio: float = 0.10) -> list[str]:
    """Rank lines by BM25 score against the task query, keep top ~10%."""
    lines = tool_output.split("\n")
    n_keep = max(1, int(len(lines) * target_ratio))

    # Simple BM25 implementation
    query_terms = set(task.lower().split())
    k1 = 1.5
    b = 0.75

    # Document frequency
    doc_freq: dict[str, int] = {}
    doc_lengths = []
    for line in lines:
        terms = set(line.lower().split())
        doc_lengths.append(len(line.lower().split()))
        for t in terms:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    n_docs = len(lines)
    avg_dl = sum(doc_lengths) / max(n_docs, 1)

    scores = []
    for i, line in enumerate(lines):
        terms = line.lower().split()
        term_freq: dict[str, int] = {}
        for t in terms:
            term_freq[t] = term_freq.get(t, 0) + 1

        score = 0.0
        dl = doc_lengths[i]
        for qt in query_terms:
            if qt not in doc_freq:
                continue
            df = doc_freq[qt]
            tf = term_freq.get(qt, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm
        scores.append((score, i))

    scores.sort(reverse=True)
    kept_indices = sorted([idx for _, idx in scores[:n_keep]])
    return [lines[i] for i in kept_indices]


# ---------------------------------------------------------------------------
# Model baselines
# ---------------------------------------------------------------------------


def _load_swe_pruner():
    """Load SWE-Pruner model (needs: pip install swe-pruner, CUDA + flash-attn)."""
    from swe_pruner.prune_wrapper import SwePrunerForCodePruning

    model = SwePrunerForCodePruning.from_pretrained("ayanami-kitasan/code-pruner")
    return model


def baseline_swe_pruner(model, task: str, tool_output: str, threshold: float = 0.5) -> list[str]:
    """Run SWE-Pruner on (task, tool_output).

    Uses PruneRequest/PruneResponse API. kept_frags is 1-indexed line numbers.
    """
    from swe_pruner.prune_wrapper import PruneRequest

    request = PruneRequest(
        query=task,
        code=tool_output,
        threshold=threshold,
        always_keep_first_frags=False,
        chunk_overlap_tokens=50,
    )
    response = model.prune(request)

    lines = tool_output.split("\n")
    kept = set(response.kept_frags)
    return [lines[i - 1] for i in sorted(kept) if 0 < i <= len(lines)]


def _load_zilliz():
    """Load Zilliz semantic-highlight (needs: pip install transformers torch)."""
    import torch
    from transformers import AutoModel

    model_name = "zilliz/semantic-highlight-bilingual-v1"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def baseline_zilliz(model, task: str, tool_output: str, threshold: float = 0.5) -> list[str]:
    """Run Zilliz semantic-highlight via get_raw_predictions().

    Uses the low-level API to avoid the broken process() path in
    transformers 5.2 (build_inputs_with_special_tokens removed).
    Each line is passed as a separate context, and per-token pruning
    probabilities are averaged per line.
    """
    import torch

    lines = tool_output.split("\n")
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return []

    with torch.no_grad():
        raw = model.get_raw_predictions(query=task, contexts=non_empty)

    kept = []
    for i, line in enumerate(non_empty):
        if i >= len(raw.context_ranges):
            break
        start, end = raw.context_ranges[i]
        segment = raw.pruning_probs[start:end]
        if segment.size > 0:
            score = float(segment.mean())
        else:
            score = 0.0
        if score >= threshold:
            kept.append(line)

    return kept


def _load_gliner2():
    """Load GLiNER2 model (needs: pip install gliner2)."""
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    return model


def baseline_gliner2(model, task: str, tool_output: str) -> list[str]:
    """Run GLiNER2 span extraction with 'relevant' as the entity label.

    Uses the task description as the label description to guide extraction.
    Extracted spans are mapped back to line numbers.
    """
    lines = tool_output.split("\n")
    if not lines:
        return []

    # Use the task as the entity description for guided extraction
    result = model.extract_entities(
        tool_output,
        {"relevant": f"Text relevant to: {task}"},
        include_spans=True,
    )

    entities = result.get("entities", {}).get("relevant", [])
    if not entities:
        return []

    # Build line offset map
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append((offset, offset + len(line)))
        offset += len(line) + 1  # +1 for newline

    # Map character spans to line indices
    kept_indices = set()
    for entity in entities:
        span_start = entity.get("start", 0)
        span_end = entity.get("end", 0)
        for i, (lo, hi) in enumerate(line_offsets):
            if span_start < hi and span_end > lo:
                kept_indices.add(i)

    return [lines[i] for i in sorted(kept_indices)]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_baseline(
    name: str,
    predict_fn,
    samples: list[dict],
    model=None,
) -> dict:
    """Run a baseline on all samples and compute aggregate metrics."""
    all_metrics = []

    for i, sample in enumerate(samples):
        task = sample["task"]
        tool_output = sample["tool_output"]
        ref_lines = [line for line in sample.get("relevant_lines", []) if line.strip()]

        if model is not None:
            pred_lines = predict_fn(model, task, tool_output)
        else:
            pred_lines = predict_fn(task, tool_output)

        pred_lines = [line.strip() for line in pred_lines if line.strip()]
        metrics = _compute_metrics(pred_lines, ref_lines, tool_output)
        all_metrics.append(metrics)

        if (i + 1) % 50 == 0:
            logger.info(f"  [{name}] {i + 1}/{len(samples)}")

    result = _aggregate(all_metrics)
    result["baseline"] = name
    return result


def print_results(results: list[dict]) -> None:
    """Print results as a markdown table."""
    print()
    print(
        "| Model | Span P | Span R | Span F1 | Exact Match | Fuzzy F1 | Partial Overlap | Empty Acc | ROUGE-L | Compression |"
    )
    print(
        "|-------|--------|--------|---------|-------------|----------|-----------------|-----------|---------|-------------|"
    )
    for r in results:
        name = r["baseline"]
        print(
            f"| {name} "
            f"| {r['span_precision']['mean']:.4f} "
            f"| {r['span_recall']['mean']:.4f} "
            f"| {r['span_f1']['mean']:.4f} "
            f"| {r['exact_match']['mean']:.4f} "
            f"| {r['fuzzy_span_f1']['mean']:.4f} "
            f"| {r['partial_overlap']['mean']:.4f} "
            f"| {r['empty_accuracy']['mean']:.4f} "
            f"| {r['rouge_l']['mean']:.4f} "
            f"| {r['compression']['mean']:.4f} |"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines on squeez test set")
    parser.add_argument("--eval-file", required=True, help="Path to encoder_test.jsonl")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["all"],
        help=f"Which baselines to run. Options: {', '.join(ALL_BASELINES)}, all, naive, models",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="eval_baselines.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)

    # Resolve baseline list
    baselines = set()
    for b in args.baselines:
        if b == "all":
            baselines.update(ALL_BASELINES)
        elif b == "naive":
            baselines.update(ALL_NAIVE)
        elif b == "models":
            baselines.update(ALL_MODEL)
        else:
            baselines.add(b)

    # Load data
    samples = []
    with open(args.eval_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info(f"Loaded {len(samples)} samples from {args.eval_file}")

    results = []

    # Naive baselines
    if "random" in baselines:
        logger.info("Running: random")
        results.append(evaluate_baseline("Random (10%)", baseline_random, samples))

    if "first_n" in baselines:
        logger.info("Running: first_n")
        results.append(evaluate_baseline("First-N (10%)", baseline_first_n, samples))

    if "last_n" in baselines:
        logger.info("Running: last_n")
        results.append(evaluate_baseline("Last-N (10%)", baseline_last_n, samples))

    if "bm25" in baselines:
        logger.info("Running: bm25")
        results.append(evaluate_baseline("BM25 (10%)", baseline_bm25, samples))

    # Model baselines
    if "swe_pruner" in baselines:
        logger.info("Loading SWE-Pruner...")
        try:
            model = _load_swe_pruner()
            logger.info("Running: swe_pruner")
            results.append(
                evaluate_baseline("SWE-Pruner", baseline_swe_pruner, samples, model=model)
            )
        except ImportError:
            logger.error("swe-pruner not installed. pip install swe-pruner")
        except Exception as e:
            logger.error(f"SWE-Pruner failed: {e}")

    if "zilliz" in baselines:
        logger.info("Loading Zilliz semantic-highlight...")
        try:
            model = _load_zilliz()
            logger.info("Running: zilliz")
            results.append(
                evaluate_baseline(
                    "Zilliz Semantic-Highlight", baseline_zilliz, samples, model=model
                )
            )
        except Exception as e:
            logger.error(f"Zilliz failed: {type(e).__name__}: {e}")

    if "gliner2" in baselines:
        logger.info("Loading GLiNER2...")
        try:
            model = _load_gliner2()
            logger.info("Running: gliner2")
            results.append(
                evaluate_baseline("GLiNER2-Large", baseline_gliner2, samples, model=model)
            )
        except ImportError:
            logger.error("gliner2 not installed. pip install gliner2")
        except Exception as e:
            logger.error(f"GLiNER2 failed: {e}")

    # Print and save
    print_results(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
