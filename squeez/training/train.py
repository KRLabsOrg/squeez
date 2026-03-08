"""LoRA fine-tuning script for squeez.

Fine-tunes Qwen 3.5 (or similar) with LoRA adapters using Unsloth + SFTTrainer.
"""

import argparse
import json
import logging
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


def _load_dataset_for_sft(data_path: str) -> list[dict]:
    """Load JSONL and concatenate prompt+response into a 'text' field."""
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                samples.append({"text": row["prompt"] + row["response"] + "<|im_end|>"})
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def train(args: argparse.Namespace):
    """Run LoRA fine-tuning with Unsloth + SFTTrainer."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    config = load_config(args.config)

    model_name = args.base_model or config.get("model", "Qwen/Qwen3.5-2B")
    max_length = args.max_length or config.get("max_length", 16384)
    batch_size = args.batch_size or config.get("batch_size", 8)
    grad_accum = args.gradient_accumulation_steps or config.get("gradient_accumulation_steps", 4)
    lr = args.lr or config.get("learning_rate", 2e-4)
    epochs = args.epochs or config.get("num_epochs", 3)
    lora_r = args.lora_r or config.get("lora_r", 16)
    lora_alpha = args.lora_alpha or config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0)
    output_dir = args.output_dir or "output/squeez_qwen"

    # 1. Load model
    logger.info(f"Loading {model_name} with Unsloth (bf16 LoRA, r={lora_r}, alpha={lora_alpha})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Qwen 3.5 returns a VLProcessor — extract the text tokenizer
    if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
        logger.info("Extracting text tokenizer from VL processor")
        tokenizer = tokenizer.tokenizer

    # 2. Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_length,
    )
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Load datasets
    train_data = _load_dataset_for_sft(args.train_file)
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = None
    if args.eval_file:
        eval_data = _load_dataset_for_sft(args.eval_file)
        eval_dataset = Dataset.from_list(eval_data)

    # 4. Configure SFTTrainer
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        max_seq_length=max_length,
        packing=False,
        dataset_text_field="text",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if eval_dataset else None,
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        logging_steps=config.get("logging_steps", 25),
        bf16=True,
        optim="adamw_8bit",
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 5. Mask prompt tokens — only train on assistant response
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    # 7. Save
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
