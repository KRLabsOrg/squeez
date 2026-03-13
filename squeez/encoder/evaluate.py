"""Evaluation script for the encoder line classifier.

Runs inference on an eval set and computes the same metrics as the generative
model's evaluate.py for direct comparison:
- Span precision/recall/F1 (line-level set overlap)
- Exact match
- Partial overlap
- Empty accuracy (correctly predicting empty vs non-empty)
- ROUGE-L
- Compression ratio

Usage:
    python -m squeez.encoder.evaluate \
        --model-path output/squeez_encoder \
        --eval-file data/encoder_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics

logger = logging.getLogger(__name__)


def evaluate_encoder(
    model_path: str,
    eval_file: str,
    max_samples: int | None = None,
    threshold: float = 0.5,
    examples_output: str | None = None,
) -> dict:
    """Evaluate the encoder model on an eval set.

    Args:
        model_path: Path to trained encoder model
        eval_file: Path to encoder-format JSONL
        max_samples: Maximum samples to evaluate
        threshold: Relevance score threshold

    Returns:
        Dict with aggregate metrics (same format as generative evaluate.py)
    """
    import torch
    from transformers import AutoTokenizer

    from squeez.encoder.model import LINE_SEP_TOKEN, SqueezEncoderForLineClassification
    from squeez.training.evaluate import (
        compute_compression_ratio,
        compute_empty_accuracy,
        compute_fuzzy_span_metrics,
        compute_partial_overlap,
        compute_rouge_l,
        compute_span_metrics,
    )

    logger.info(f"Loading encoder model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure LINE_SEP is in tokenizer
    if tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [LINE_SEP_TOKEN]})

    model = SqueezEncoderForLineClassification.from_pretrained(model_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load eval data
    samples = []
    with open(eval_file) as f:
        for line in f:
            if line.strip():
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

    for i, sample in enumerate(samples):
        task = sample["task"]
        tool_output = sample["tool_output"]
        ref_lines = [line for line in sample.get("relevant_lines", []) if line.strip()]

        # Run encoder inference
        pred_lines = model.extract(
            task=task,
            tool_output=tool_output,
            tokenizer=tokenizer,
            threshold=threshold,
        )
        # Strip for fair comparison
        pred_lines = [line.strip() for line in pred_lines if line.strip()]

        # Span metrics
        span = compute_span_metrics(pred_lines, ref_lines)
        all_metrics["span_precision"].append(span["precision"])
        all_metrics["span_recall"].append(span["recall"])
        all_metrics["span_f1"].append(span["f1"])
        all_metrics["exact_match"].append(span["exact_match"])

        fuzzy = compute_fuzzy_span_metrics(pred_lines, ref_lines, threshold=0.5)
        all_metrics["fuzzy_span_precision"].append(fuzzy["precision"])
        all_metrics["fuzzy_span_recall"].append(fuzzy["recall"])
        all_metrics["fuzzy_span_f1"].append(fuzzy["f1"])

        # Partial overlap
        partial = compute_partial_overlap(pred_lines, ref_lines)
        all_metrics["partial_overlap"].append(partial)

        # Empty accuracy
        empty = compute_empty_accuracy(pred_lines, ref_lines)
        all_metrics["empty_accuracy"].append(empty["correct"])
        empty_confusion[empty["category"]] += 1

        # ROUGE-L on concatenated text
        pred_text = "\n".join(pred_lines)
        ref_text = "\n".join(ref_lines)
        rouge = compute_rouge_l(pred_text, ref_text)
        all_metrics["rouge_l"].append(rouge)

        # Compression
        compression = compute_compression_ratio(tool_output, pred_text)
        all_metrics["compression"].append(compression)

        examples.append(
            {
                "task": task,
                "tool_output": tool_output,
                "predicted_lines": pred_lines,
                "reference_lines": ref_lines,
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

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{len(samples)}] "
                f"F1={span['f1']:.3f} EM={span['exact_match']:.0f} "
                f"ROUGE-L={rouge:.3f}"
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
    results["model_type"] = "encoder"
    results["threshold"] = threshold

    if examples_output:
        with open(examples_output, "w") as f:
            json.dump(examples, f, indent=2)
        logger.info(f"Saved per-sample examples to {examples_output}")

    logger.info("=" * 60)
    logger.info("ENCODER EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, stats in results.items():
        if isinstance(stats, dict) and "mean" in stats:
            logger.info(f"  {key:20s}: mean={stats['mean']:.4f}  median={stats['median']:.4f}")
    logger.info(f"  {'empty_confusion':20s}: {empty_confusion}")

    return results


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate squeez encoder model")

    parser.add_argument("--model-path", required=True, help="Path to trained encoder model")
    parser.add_argument("--eval-file", required=True, help="Path to encoder-format eval JSONL")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="eval_results_encoder.json", help="Output results file")
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

    results = evaluate_encoder(
        args.model_path,
        args.eval_file,
        args.max_samples,
        args.threshold,
        args.examples_output,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
