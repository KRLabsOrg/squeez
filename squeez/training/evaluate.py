"""Evaluation script for tool output extraction model.

Metrics:
- Line-level Precision/Recall/F1 (vs ground truth relevant lines)
- ROUGE-L between generated and reference filtered output
- Compression ratio (output tokens / input tokens)
"""

import argparse
import json
import logging
import re
import statistics

logger = logging.getLogger(__name__)


def extract_line_numbers(text: str) -> set[int]:
    """Extract line numbers from filtered output."""
    line_nums = set()
    for line in text.split("\n"):
        match = re.match(r"^(\d+):", line)
        if match:
            line_nums.add(int(match.group(1)))
    return line_nums


def compute_line_level_metrics(predicted: str, reference: str) -> dict[str, float]:
    """Compute line-level precision, recall, and F1."""
    pred_lines = extract_line_numbers(predicted)
    ref_lines = extract_line_numbers(reference)

    if not ref_lines:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_lines & ref_lines)
    precision = tp / len(pred_lines) if pred_lines else 0.0
    recall = tp / len(ref_lines) if ref_lines else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


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
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0

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


def evaluate_model(
    model_path: str,
    eval_file: str,
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
) -> dict:
    """Evaluate the model on the eval set.

    Args:
        model_path: Path to trained model
        eval_file: Path to eval.jsonl
        max_samples: Maximum samples to evaluate
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with aggregate metrics

    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load eval data
    samples = []
    with open(eval_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Evaluating on {len(samples)} samples")

    all_metrics = {
        "line_precision": [],
        "line_recall": [],
        "line_f1": [],
        "rouge_l": [],
        "compression": [],
    }

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        reference = sample["response"]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Compute metrics
        line_metrics = compute_line_level_metrics(generated, reference)
        rouge = compute_rouge_l(generated, reference)
        original_output = prompt.split("<|user|>")[-1].split("<|assistant|>")[0]
        compression = compute_compression_ratio(original_output, generated)

        all_metrics["line_precision"].append(line_metrics["precision"])
        all_metrics["line_recall"].append(line_metrics["recall"])
        all_metrics["line_f1"].append(line_metrics["f1"])
        all_metrics["rouge_l"].append(rouge)
        all_metrics["compression"].append(compression)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{len(samples)}] F1={line_metrics['f1']:.3f} ROUGE-L={rouge:.3f}"
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

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for key, stats in results.items():
        logger.info(f"  {key}: mean={stats['mean']:.4f} median={stats['median']:.4f}")

    return results


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the parser for the evaluation CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate tool output extractor")

    parser.add_argument(
        "--extractor-model",
        "--model-path",
        dest="extractor_model",
        required=True,
        help="Path to the trained extractor model",
    )
    parser.add_argument("--eval-file", required=True, help="Path to eval.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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
    )

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to eval_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
