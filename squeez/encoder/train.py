"""Training script for the encoder line classifier.

Supports two classifier types:
- token: Token-level classification with [LINE_SEP] markers (original)
- sentence: Sentence-level per-line classification with Unsloth (faster)

Usage:
    # Token classifier (original)
    python -m squeez.encoder.train \
        --train-file data/encoder_train.jsonl \
        --eval-file data/encoder_dev.jsonl \
        --base-model jhu-clsp/mmBERT-base \
        --output-dir output/squeez_encoder

    # Sentence classifier with Unsloth
    python -m squeez.encoder.train \
        --classifier-type sentence \
        --train-file data/encoder_train.jsonl \
        --eval-file data/encoder_dev.jsonl \
        --base-model unsloth/ModernBERT-large \
        --output-dir output/squeez_sentence
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
    max_negative_ratio: float | None = None,
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
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    # Load datasets
    logger.info(f"Loading train data from {train_file}")
    train_dataset = LineClassificationDataset(
        train_file, tokenizer, max_length, max_negative_ratio=max_negative_ratio
    )
    eval_dataset = None
    if eval_file:
        logger.info(f"Loading eval data from {eval_file}")
        eval_dataset = LineClassificationDataset(eval_file, tokenizer, max_length)

    # Data collator (handles dynamic padding for token classification)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
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


def train_pooled(
    train_file: str,
    eval_file: str | None,
    base_model: str = "answerdotai/ModernBERT-large",
    output_dir: str = "output/squeez_pooled",
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
    max_negative_ratio: float | None = None,
) -> None:
    """Train the pooled line classifier: single-pass encoder + line-level mean-pool."""
    import torch
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from squeez.encoder.model import LINE_SEP_TOKEN
    from squeez.encoder.sentence import (
        PooledLineClassifier,
        PooledLineConfig,
        PooledLineDataset,
        collate_pooled_lines,
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

    # Create model
    logger.info(f"Creating pooled line classifier with base: {base_model}")
    config = PooledLineConfig(
        base_model_name=base_model,
        vocab_size=len(tokenizer),
        max_length=max_length,
    )
    model = PooledLineClassifier.from_encoder_pretrained(config)

    model.encoder.resize_token_embeddings(len(tokenizer))
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    # Store token IDs the model needs during forward pass
    model._line_sep_id = tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN)
    model._sep_token_id = tokenizer.sep_token_id

    # Load datasets
    logger.info(f"Loading train data from {train_file}")
    train_dataset = PooledLineDataset(
        train_file, tokenizer, max_length, max_negative_ratio=max_negative_ratio
    )
    eval_dataset = None
    if eval_file:
        logger.info(f"Loading eval data from {eval_file}")
        eval_dataset = PooledLineDataset(eval_file, tokenizer, max_length)

    # Custom trainer that injects token IDs into forward kwargs
    # and saves tokenizer with each checkpoint
    class PooledTrainer(Trainer):
        def __init__(self, *args, tokenizer_to_save=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._tokenizer_to_save = tokenizer_to_save

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs["line_sep_id"] = model._line_sep_id
            inputs["sep_token_id"] = model._sep_token_id
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics=metrics)
            if self._tokenizer_to_save is not None:
                checkpoint_dir = self.state.best_model_checkpoint or (
                    Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                )
                self._tokenizer_to_save.save_pretrained(checkpoint_dir)

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
        remove_unused_columns=False,
    )

    trainer = PooledTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_pooled_lines,
        tokenizer_to_save=tokenizer,
    )

    logger.info("Starting pooled line classifier training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy standalone modeling file for trust_remote_code support
    import json
    import shutil

    standalone_src = Path(__file__).parent / "modeling_squeez_pooled.py"
    standalone_dst = Path(output_dir) / "modeling_squeez_pooled.py"
    if standalone_src.exists():
        shutil.copy(standalone_src, standalone_dst)
        logger.info(f"Copied standalone modeling file to {standalone_dst}")

    # Set auto_map in config for AutoModel.from_pretrained(..., trust_remote_code=True)
    config_path = Path(output_dir) / "config.json"
    with open(config_path) as f:
        saved_config = json.load(f)
    saved_config["auto_map"] = {
        "AutoConfig": "modeling_squeez_pooled.PooledLineConfig",
        "AutoModel": "modeling_squeez_pooled.PooledLineClassifier",
    }
    saved_config["model_type"] = "squeez-pooled"
    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)

    logger.info("Training complete.")
    logger.info(f"To clean up checkpoints before uploading: rm -rf {output_dir}/checkpoint-*")


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train squeez encoder line classifier")

    parser.add_argument(
        "--classifier-type",
        choices=["token", "pooled"],
        default=None,
        help="Classifier type: token (per-token classification) or pooled (line-level mean-pool)",
    )
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
    parser.add_argument(
        "--max-negative-ratio",
        type=float,
        default=None,
        help="Cap fraction of all-negative windows in training data (e.g. 0.40)",
    )
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

    classifier_type = args.classifier_type or config.get("encoder_classifier_type", "token")

    if classifier_type == "pooled":
        train_pooled(
            train_file=args.train_file,
            eval_file=args.eval_file,
            base_model=args.base_model
            or config.get("pooled_base_model", "answerdotai/ModernBERT-large"),
            output_dir=args.output_dir or config.get("pooled_output_dir", "output/squeez_pooled"),
            max_length=args.max_length or config.get("pooled_max_length", 8192),
            batch_size=args.batch_size or config.get("pooled_batch_size", 2),
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
                or config.get("pooled_gradient_accumulation_steps", 8)
            ),
            learning_rate=args.learning_rate or config.get("pooled_learning_rate", 2e-5),
            num_epochs=args.num_epochs or config.get("pooled_num_epochs", 5),
            warmup_ratio=args.warmup_ratio or config.get("pooled_warmup_ratio", 0.1),
            weight_decay=args.weight_decay or config.get("weight_decay", 0.01),
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            eval_batch_size=args.eval_batch_size or config.get("pooled_eval_batch_size", 1),
            eval_accumulation_steps=(
                args.eval_accumulation_steps or config.get("pooled_eval_accumulation_steps", 1)
            ),
            max_negative_ratio=(
                args.max_negative_ratio
                if args.max_negative_ratio is not None
                else config.get("pooled_max_negative_ratio")
            ),
        )
    else:
        train(
            train_file=args.train_file,
            eval_file=args.eval_file,
            base_model=args.base_model or config.get("encoder_base_model", "jhu-clsp/mmBERT-base"),
            output_dir=args.output_dir or config.get("encoder_output_dir", "output/squeez_encoder"),
            max_length=args.max_length or config.get("encoder_max_length", 8192),
            batch_size=args.batch_size or config.get("encoder_batch_size", 2),
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
                or config.get("encoder_gradient_accumulation_steps", 8)
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
            max_negative_ratio=(
                args.max_negative_ratio
                if args.max_negative_ratio is not None
                else config.get("encoder_max_negative_ratio")
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
