"""Evaluation script for squeez tool output extraction model.

Metrics:
- Span Exact Match: fraction of samples where predicted lines == reference lines exactly
- Span Precision/Recall/F1: line-level set overlap between predicted and reference
- Empty Accuracy: correctly predicting empty vs non-empty relevant_lines
- ROUGE-L: token-level overlap between concatenated predicted and reference lines
- Compression ratio: output lines / input lines
"""

import argparse
import concurrent.futures
import json
import logging
import re
import statistics

logger = logging.getLogger(__name__)


def _parse_relevant_lines(text: str) -> list[str]:
    """Parse relevant_lines from model output (JSON or raw text).

    Handles:
    - Valid JSON: {"relevant_lines": ["line1", "line2"]}
    - Raw text fallback: split by newlines
    """
    text = text.strip()
    try:
        data = json.loads(text)
        lines = data.get("relevant_lines", [])
        if isinstance(lines, list):
            return [str(line).strip() for line in lines if str(line).strip()]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Fallback: treat each non-empty line as a span
    return [line.strip() for line in text.split("\n") if line.strip()]


def compute_span_metrics(predicted: list[str], reference: list[str]) -> dict[str, float]:
    """Compute span-level precision, recall, F1 using set overlap on normalized lines."""
    pred_set = set(predicted)
    ref_set = set(reference)

    if not ref_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}

    if not ref_set or not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set)
    recall = tp / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred_set == ref_set else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
    }


def compute_partial_overlap(predicted: list[str], reference: list[str]) -> float:
    """Compute partial overlap ratio using character-level intersection.

    For each reference line, find the best matching predicted line (substring match)
    and compute the fraction of reference characters covered.
    """
    if not reference:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0

    total_chars = 0
    matched_chars = 0

    for ref_line in reference:
        total_chars += len(ref_line)
        best = 0
        for pred_line in predicted:
            # Check substring containment both ways
            if ref_line in pred_line or pred_line in ref_line:
                best = max(best, min(len(ref_line), len(pred_line)))
            else:
                # Character-level overlap via set intersection on character bigrams
                ref_bigrams = (
                    {ref_line[i : i + 2] for i in range(len(ref_line) - 1)}
                    if len(ref_line) > 1
                    else {ref_line}
                )
                pred_bigrams = (
                    {pred_line[i : i + 2] for i in range(len(pred_line) - 1)}
                    if len(pred_line) > 1
                    else {pred_line}
                )
                if ref_bigrams:
                    overlap = len(ref_bigrams & pred_bigrams) / len(ref_bigrams)
                    best = max(best, int(overlap * len(ref_line)))
        matched_chars += best

    return round(matched_chars / total_chars, 4) if total_chars > 0 else 0.0


def _line_overlap_score(pred_line: str, ref_line: str) -> float:
    """Compute a symmetric overlap score for two lines."""
    if not pred_line or not ref_line:
        return 0.0
    if pred_line == ref_line:
        return 1.0
    if pred_line in ref_line or ref_line in pred_line:
        return min(len(pred_line), len(ref_line)) / max(len(pred_line), len(ref_line))

    pred_bigrams = (
        {pred_line[i : i + 2] for i in range(len(pred_line) - 1)}
        if len(pred_line) > 1
        else {pred_line}
    )
    ref_bigrams = (
        {ref_line[i : i + 2] for i in range(len(ref_line) - 1)} if len(ref_line) > 1 else {ref_line}
    )
    if not pred_bigrams or not ref_bigrams:
        return 0.0

    overlap = len(pred_bigrams & ref_bigrams)
    precision = overlap / len(pred_bigrams)
    recall = overlap / len(ref_bigrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_fuzzy_span_metrics(
    predicted: list[str],
    reference: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute one-to-one fuzzy line overlap metrics at a fixed threshold."""
    if not reference and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not reference or not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_idx, pred_line in enumerate(predicted):
        for ref_idx, ref_line in enumerate(reference):
            score = _line_overlap_score(pred_line, ref_line)
            if score >= threshold:
                candidate_pairs.append((score, pred_idx, ref_idx))

    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    tp = 0

    for score, pred_idx, ref_idx in sorted(candidate_pairs, reverse=True):
        del score
        if pred_idx in matched_pred or ref_idx in matched_ref:
            continue
        matched_pred.add(pred_idx)
        matched_ref.add(ref_idx)
        tp += 1

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_empty_accuracy(predicted: list[str], reference: list[str]) -> dict[str, float | str]:
    """Check if model correctly predicts empty vs non-empty.

    Returns category (true_positive, true_negative, false_positive, false_negative)
    and whether correct.
    """
    ref_empty = len(reference) == 0
    pred_empty = len(predicted) == 0

    if ref_empty and pred_empty:
        return {"category": "true_negative", "correct": 1.0}
    elif ref_empty and not pred_empty:
        return {"category": "false_positive", "correct": 0.0}
    elif not ref_empty and pred_empty:
        return {"category": "false_negative", "correct": 0.0}
    else:
        return {"category": "true_positive", "correct": 1.0}


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute longest common subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(predicted: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    pred_tokens = predicted.split()
    ref_tokens = reference.split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_compression_ratio(original: str, filtered: str) -> float:
    """Compute compression ratio (1 - filtered/original)."""
    orig_lines = len(original.split("\n"))
    filt_lines = len(filtered.split("\n"))
    if orig_lines == 0:
        return 0.0
    return round(1.0 - filt_lines / orig_lines, 4)


def _parse_prompt_sections(prompt: str) -> tuple[str, str]:
    """Extract task and tool output from the ChatML prompt."""
    task_match = re.search(r"<task>\n(.*?)\n</task>", prompt, re.DOTALL)
    output_match = re.search(r"<tool_output>\n(.*?)\n</tool_output>", prompt, re.DOTALL)
    task = task_match.group(1) if task_match else ""
    tool_output = output_match.group(1) if output_match else ""
    return task, tool_output


def evaluate_model(
    model_path: str | None,
    eval_file: str,
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
    server_url: str | None = None,
    server_model: str | None = None,
    temperature: float = 0.1,
    request_concurrency: int = 1,
    examples_output: str | None = None,
) -> dict:
    """Evaluate the model on the eval set.

    Args:
        model_path: Path to trained model, if evaluating locally
        eval_file: Path to eval.jsonl
        max_samples: Maximum samples to evaluate
        max_new_tokens: Max tokens to generate
        server_url: OpenAI-compatible server URL, if evaluating remotely
        server_model: Remote model ID for the server backend
        temperature: Generation temperature
        request_concurrency: Number of concurrent remote requests for server evaluation
        examples_output: Optional path to save per-sample predictions/metrics

    Returns:
        Dict with aggregate metrics
    """
    from squeez.inference.extractor import ToolOutputExtractor

    if not model_path and not server_url:
        raise ValueError(
            "Pass either model_path for local evaluation or server_url for remote evaluation."
        )

    target = server_url or model_path
    logger.info(f"Loading extractor from {target}")
    extractor = ToolOutputExtractor(
        model_path=model_path,
        base_url=server_url,
        model_name=server_model,
    )

    # Load eval data
    samples = []
    with open(eval_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Evaluating on {len(samples)} samples")

    all_metrics = {
        "span_precision": [],
        "span_recall": [],
        "span_f1": [],
        "exact_match": [],
        "fuzzy_span_precision": [],
        "fuzzy_span_recall": [],
        "fuzzy_span_f1": [],
        "partial_overlap": [],
        "empty_accuracy": [],
        "rouge_l": [],
        "compression": [],
    }

    empty_confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    examples: list[dict] = []
    num_errors = 0

    def evaluate_sample(sample: dict) -> dict:
        prompt = sample["prompt"]
        reference_raw = sample["response"]
        task, tool_output = _parse_prompt_sections(prompt)
        try:
            generated_raw = extractor.extract(
                task=task,
                tool_output=tool_output,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - remote backends are integration-level
            return {
                "prompt": prompt,
                "task": task,
                "tool_output": tool_output,
                "pred_lines": [],
                "ref_lines": _parse_relevant_lines(reference_raw),
                "error": f"{exc.__class__.__name__}: {exc}",
            }

        # Parse both into line lists
        pred_lines = _parse_relevant_lines(generated_raw)
        ref_lines = _parse_relevant_lines(reference_raw)

        # Span metrics
        span = compute_span_metrics(pred_lines, ref_lines)
        fuzzy = compute_fuzzy_span_metrics(pred_lines, ref_lines, threshold=0.5)

        # Partial overlap
        partial = compute_partial_overlap(pred_lines, ref_lines)

        # Empty accuracy
        empty = compute_empty_accuracy(pred_lines, ref_lines)

        # ROUGE-L on concatenated text
        pred_text = "\n".join(pred_lines)
        ref_text = "\n".join(ref_lines)
        rouge = compute_rouge_l(pred_text, ref_text)

        # Compression
        compression = compute_compression_ratio(tool_output, pred_text)
        return {
            "prompt": prompt,
            "task": task,
            "tool_output": tool_output,
            "pred_lines": pred_lines,
            "ref_lines": ref_lines,
            "span": span,
            "fuzzy": fuzzy,
            "partial": partial,
            "empty": empty,
            "rouge": rouge,
            "compression": compression,
        }

    def record_result(result: dict) -> None:
        nonlocal num_errors
        if "error" in result:
            num_errors += 1
            examples.append(
                {
                    "prompt": result["prompt"],
                    "task": result["task"],
                    "tool_output": result["tool_output"],
                    "predicted_lines": result["pred_lines"],
                    "reference_lines": result["ref_lines"],
                    "error": result["error"],
                }
            )
            logger.warning("Sample evaluation failed: %s", result["error"])
            return

        span = result["span"]
        fuzzy = result["fuzzy"]
        partial = result["partial"]
        empty = result["empty"]
        rouge = result["rouge"]
        compression = result["compression"]

        all_metrics["span_precision"].append(span["precision"])
        all_metrics["span_recall"].append(span["recall"])
        all_metrics["span_f1"].append(span["f1"])
        all_metrics["exact_match"].append(span["exact_match"])
        all_metrics["fuzzy_span_precision"].append(fuzzy["precision"])
        all_metrics["fuzzy_span_recall"].append(fuzzy["recall"])
        all_metrics["fuzzy_span_f1"].append(fuzzy["f1"])
        all_metrics["partial_overlap"].append(partial)
        all_metrics["empty_accuracy"].append(empty["correct"])
        empty_confusion[empty["category"]] += 1
        all_metrics["rouge_l"].append(rouge)
        all_metrics["compression"].append(compression)
        examples.append(
            {
                "prompt": result["prompt"],
                "task": result["task"],
                "tool_output": result["tool_output"],
                "predicted_lines": result["pred_lines"],
                "reference_lines": result["ref_lines"],
                "metrics": {
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
                },
            }
        )

    use_concurrency = bool(server_url) and request_concurrency > 1

    if use_concurrency:
        logger.info(f"Using request concurrency={request_concurrency} for remote evaluation")
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_concurrency) as pool:
            futures = [pool.submit(evaluate_sample, sample) for sample in samples]
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                record_result(future.result())

                if i % 10 == 0:
                    logger.info(
                        f"  [{i}/{len(samples)}] "
                        f"F1={all_metrics['span_f1'][-1]:.3f} "
                        f"EM={all_metrics['exact_match'][-1]:.0f} "
                        f"ROUGE-L={all_metrics['rouge_l'][-1]:.3f}"
                    )
    else:
        for i, sample in enumerate(samples):
            result = evaluate_sample(sample)
            record_result(result)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{i + 1}/{len(samples)}] "
                    f"F1={result['span']['f1']:.3f} EM={result['span']['exact_match']:.0f} "
                    f"ROUGE-L={result['rouge']:.3f}"
                )

    # Aggregate
    results = {}
    for key, values in all_metrics.items():
        if values:
            results[key] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            }

    results["empty_confusion"] = empty_confusion
    results["num_samples"] = len(samples)
    results["num_errors"] = num_errors

    if examples_output:
        with open(examples_output, "w") as f:
            json.dump(examples, f, indent=2)
        logger.info(f"Saved per-sample examples to {examples_output}")

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, stats in results.items():
        if isinstance(stats, dict) and "mean" in stats:
            logger.info(f"  {key:20s}: mean={stats['mean']:.4f}  median={stats['median']:.4f}")
    logger.info(f"  {'empty_confusion':20s}: {empty_confusion}")

    return results


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the parser for the evaluation CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate tool output extractor")

    parser.add_argument(
        "--extractor-model",
        "--model-path",
        dest="extractor_model",
        default=None,
        help="Path to the trained extractor model",
    )
    parser.add_argument(
        "--server-url",
        "--base-url",
        dest="server_url",
        default=None,
        help="URL for an OpenAI-compatible model server",
    )
    parser.add_argument(
        "--server-model",
        "--model-name",
        dest="server_model",
        default=None,
        help="Model ID on the remote server (auto-detected if omitted)",
    )
    parser.add_argument("--eval-file", required=True, help="Path to test.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=1,
        help="Concurrent requests for remote server evaluation",
    )
    parser.add_argument(
        "--examples-output",
        default=None,
        help="Optional JSON file for per-sample predictions and metrics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = evaluate_model(
        args.extractor_model,
        args.eval_file,
        args.max_samples,
        args.max_new_tokens,
        args.server_url,
        args.server_model,
        args.temperature,
        args.request_concurrency,
        args.examples_output,
    )

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to eval_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
