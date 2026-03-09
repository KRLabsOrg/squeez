"""Training script for the encoder line classifier.

Uses HuggingFace Trainer with DataCollatorForTokenClassification.

Usage:
    python -m squeez.encoder.train \
        --train-file data/encoder_train.jsonl \
        --eval-file data/encoder_dev.jsonl \
        --base-model jhu-clsp/mmBERT-base \
        --output-dir output/squeez_encoder
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train(
    train_file: str,
    eval_file: str | None,
    base_model: str = "jhu-clsp/mmBERT-base",
    output_dir: str = "output/squeez_encoder",
    max_length: int = 8192,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    eval_steps: int = 200,
    save_steps: int = 200,
    logging_steps: int = 25,
    save_total_limit: int = 3,
    fp16: bool = False,
    bf16: bool = False,
    eval_batch_size: int | None = None,
    eval_accumulation_steps: int = 1,
) -> None:
    """Train the encoder line classifier."""
    import torch
    from transformers import (
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    from squeez.encoder.dataset import LineClassificationDataset
    from squeez.encoder.model import (
        LINE_SEP_TOKEN,
        SqueezEncoderConfig,
        SqueezEncoderForLineClassification,
    )

    # Determine precision
    if not fp16 and not bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            bf16 = True
        elif torch.cuda.is_available():
            fp16 = True

    # Load tokenizer and add [LINE_SEP]
    logger.info(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [LINE_SEP_TOKEN]})
    logger.info(f"Added {num_added} special tokens: {LINE_SEP_TOKEN}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model with pretrained encoder weights
    logger.info(f"Creating encoder model with base: {base_model}")
    config = SqueezEncoderConfig(
        base_model_name=base_model,
        vocab_size=len(tokenizer),
        max_length=max_length,
    )
    model = SqueezEncoderForLineClassification.from_encoder_pretrained(config)

    # Resize embeddings for the new [LINE_SEP] token
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    # Load datasets
    logger.info(f"Loading train data from {train_file}")
    train_dataset = LineClassificationDataset(train_file, tokenizer, max_length)
    eval_dataset = None
    if eval_file:
        logger.info(f"Loading eval data from {eval_file}")
        eval_dataset = LineClassificationDataset(eval_file, tokenizer, max_length)

    # Data collator (handles dynamic padding for token classification)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size or max(1, batch_size // 2),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",
        dataloader_num_workers=0,
        eval_accumulation_steps=eval_accumulation_steps,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model + tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save auto_map in config for from_pretrained() support
    config_path = Path(output_dir) / "config.json"
    import json

    with open(config_path) as f:
        saved_config = json.load(f)
    saved_config["auto_map"] = {"AutoModel": "model.SqueezEncoderForLineClassification"}
    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)

    logger.info("Training complete.")


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train squeez encoder line classifier")

    parser.add_argument("--train-file", required=True, help="Path to encoder_train.jsonl")
    parser.add_argument("--eval-file", default=None, help="Path to encoder_dev.jsonl")
    parser.add_argument("--base-model", default=None, help="Base encoder model")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--eval-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import yaml

    default_config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    config = {}
    if default_config_path.exists():
        with open(default_config_path) as f:
            config = yaml.safe_load(f) or {}

    train(
        train_file=args.train_file,
        eval_file=args.eval_file,
        base_model=args.base_model or config.get("encoder_base_model", "jhu-clsp/mmBERT-base"),
        output_dir=args.output_dir or config.get("encoder_output_dir", "output/squeez_encoder"),
        max_length=args.max_length or config.get("encoder_max_length", 8192),
        batch_size=args.batch_size or config.get("encoder_batch_size", 2),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps or config.get("encoder_gradient_accumulation_steps", 8)
        ),
        learning_rate=args.learning_rate or config.get("encoder_learning_rate", 2e-5),
        num_epochs=args.num_epochs or config.get("encoder_num_epochs", 5),
        warmup_ratio=args.warmup_ratio or config.get("encoder_warmup_ratio", 0.1),
        weight_decay=args.weight_decay or config.get("weight_decay", 0.01),
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_batch_size=args.eval_batch_size or config.get("encoder_eval_batch_size", 1),
        eval_accumulation_steps=(
            args.eval_accumulation_steps or config.get("encoder_eval_accumulation_steps", 1)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
