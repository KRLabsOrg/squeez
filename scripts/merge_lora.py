"""Merge a PEFT/LoRA checkpoint into its base causal LM.

Example:
    python scripts/merge_lora.py \
        --base-model Qwen/Qwen3.5-2B \
        --adapter-path output/squeez_qwen/checkpoint-800 \
        --output-dir output/squeez_qwen_merged
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged model")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Torch dtype to load the base model with before merging",
    )
    return parser


def _resolve_dtype(dtype_name: str):
    import torch

    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(args.dtype)

    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    logger.info("Loading base model from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    logger.info("Loading adapter from %s", args.adapter_path)
    model = PeftModel.from_pretrained(model, args.adapter_path)

    logger.info("Merging adapter into base model")
    model = model.merge_and_unload()

    logger.info("Saving merged model to %s", args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Merge complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
