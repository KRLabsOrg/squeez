"""Merge a LoRA checkpoint into a standalone model.

Uses Unsloth for merging (same as training) to handle its internal patches
correctly, and reproduces the exact tokenizer setup from training.

Usage:
    python scripts/merge_lora.py \
        --checkpoint output/squeez_qwen/checkpoint-500 \
        --output output/squeez_qwen_merged

    # With explicit base model (if not auto-detected from adapter_config):
    python scripts/merge_lora.py \
        --checkpoint output/squeez_qwen \
        --output output/squeez_qwen_merged \
        --base-model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into standalone model")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--base-model", default=None, help="Base model (auto-detected if omitted)")
    parser.add_argument("--config", default=None, help="YAML config file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from unsloth import FastLanguageModel

    from squeez.training.train import _prepare_text_tokenizer, load_config

    config = load_config(args.config)
    max_length = config.get("max_length", 16384)

    # Detect base model from adapter config if not provided
    base_model_name = args.base_model
    adapter_config_path = Path(args.checkpoint) / "adapter_config.json"
    if not base_model_name and adapter_config_path.exists():
        with open(adapter_config_path) as f:
            base_model_name = json.load(f).get("base_model_name_or_path", "")
    if not base_model_name:
        base_model_name = config.get("model", "Qwen/Qwen3.5-2B")

    logger.info(f"Loading checkpoint from {args.checkpoint} (base: {base_model_name})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.checkpoint,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )

    # Reproduce the same tokenizer patches as training
    tokenizer = _prepare_text_tokenizer(base_model_name, tokenizer)

    logger.info(f"Merging and saving to {args.output}")
    model.save_pretrained_merged(
        args.output,
        tokenizer,
        save_method="merged_16bit",
    )

    # Copy VL processor files from base model if needed (e.g. preprocessor_config.json).
    # Unsloth saves the VL architecture in config.json but _prepare_text_tokenizer
    # strips the processor, so these files are missing from the merged output.
    config_path = Path(args.output) / "config.json"
    with open(config_path) as f:
        saved_config = json.load(f)
    archs = saved_config.get("architectures", [])
    is_vl = any("ConditionalGeneration" in a or "VL" in a for a in archs)

    if is_vl:
        import shutil

        from huggingface_hub import hf_hub_download

        vl_files = ["preprocessor_config.json", "chat_template.json"]
        for filename in vl_files:
            dest = Path(args.output) / filename
            if dest.exists():
                continue
            try:
                src = hf_hub_download(base_model_name, filename)
                shutil.copy(src, dest)
                logger.info(f"Copied {filename} from {base_model_name}")
            except Exception:
                pass  # File may not exist for all models

    logger.info(f"Done. Merged model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
