"""LoRA fine-tuning script for squeez.

Fine-tunes Qwen 3.5 (or similar) with LoRA adapters.
Uses Unsloth for optimized training when available, falls back to HF Trainer.
"""

import argparse
import logging
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> dict:
    """Load training config from YAML file."""
    import yaml

    default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def _try_unsloth():
    """Check if Unsloth is available."""
    try:
        from unsloth import FastModel

        return FastModel
    except ImportError:
        return None


def _load_model_unsloth(FastModel, model_name, max_length, lora_r, lora_alpha, lora_dropout):
    """Load model and tokenizer using Unsloth."""
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_length,
    )

    return model, tokenizer


def _load_model_transformers(model_name, lora_r, lora_alpha, lora_dropout):
    """Load model and tokenizer using vanilla transformers + peft."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def train(args: argparse.Namespace):
    """Run LoRA fine-tuning."""
    import torch
    from transformers import Trainer, TrainingArguments

    from squeez.training.dataset import ExtractionSFTDataset, collate_fn

    config = load_config(args.config)

    # Resolve parameters (CLI args override config)
    model_name = args.base_model or config.get("model", "Qwen/Qwen3.5-2B")
    max_length = args.max_length or config.get("max_length", 16384)
    batch_size = args.batch_size or config.get("batch_size", 8)
    grad_accum = args.gradient_accumulation_steps or config.get("gradient_accumulation_steps", 4)
    lr = args.lr or config.get("learning_rate", 2e-4)
    epochs = args.epochs or config.get("num_epochs", 3)
    lora_r = args.lora_r or config.get("lora_r", 16)
    lora_alpha = args.lora_alpha or config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.05)
    output_dir = args.output_dir or "output/squeez_qwen"

    # Load model — prefer Unsloth if available
    FastModel = _try_unsloth()
    if FastModel and not args.no_unsloth:
        logger.info(f"Loading {model_name} with Unsloth (bf16 LoRA, r={lora_r}, alpha={lora_alpha})")
        model, tokenizer = _load_model_unsloth(
            FastModel, model_name, max_length, lora_r, lora_alpha, lora_dropout
        )
    else:
        if not args.no_unsloth:
            logger.warning("Unsloth not installed, falling back to transformers. "
                           "Install with: pip install unsloth")
        logger.info(f"Loading {model_name} with transformers (r={lora_r}, alpha={lora_alpha})")
        model, tokenizer = _load_model_transformers(
            model_name, lora_r, lora_alpha, lora_dropout
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.print_trainable_parameters()

    # Load datasets
    train_dataset = ExtractionSFTDataset(args.train_file, tokenizer, max_length)
    eval_dataset = None
    if args.eval_file:
        eval_dataset = ExtractionSFTDataset(args.eval_file, tokenizer, max_length)

    # Training arguments
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if eval_dataset else None,
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        logging_steps=config.get("logging_steps", 25),
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete!")


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the parser for the training CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Train tool output extractor with LoRA")

    parser.add_argument("--train-file", required=True, help="Path to train.jsonl")
    parser.add_argument("--eval-file", default=None, help="Path to dev.jsonl")
    parser.add_argument(
        "--base-model",
        "--model",
        dest="base_model",
        default=None,
        help="Base model name or path to fine-tune",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--no-unsloth", action="store_true", help="Disable Unsloth even if installed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
