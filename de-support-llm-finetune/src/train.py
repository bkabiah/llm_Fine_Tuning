"""
LoRA / QLoRA fine-tuning of a small instruction-tuned LLM on the synthetic
German customer-support dataset, using Hugging Face `transformers`, `peft`,
and `trl`'s SFTTrainer.

Example:
    python src/train.py \
        --config configs/training_config.yaml \
        --lora_config configs/lora_config.yaml \
        --output_dir outputs/qwen2.5-1.5b-support-lora
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_yaml  # noqa: E402
from prepare_dataset import build_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--lora_config", default="configs/lora_config.yaml")
    parser.add_argument("--output_dir", default="outputs/lora-adapter")
    # CLI overrides for the most commonly tweaked values
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--load_in_4bit", action="store_true", default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if cfg.get("bf16", True) and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    lora_cfg_dict = load_yaml(args.lora_config)

    # apply CLI overrides
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.load_in_4bit:
        cfg["load_in_4bit"] = True
    if args.num_train_epochs is not None:
        cfg["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate

    train_path = Path(cfg["train_file"])
    val_path = Path(cfg["val_file"])
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "Training/validation data not found. Run "
            "`python data/generate_synthetic_data.py` first."
        )

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_dataset: Dataset = build_dataset(str(train_path), tokenizer)
    eval_dataset: Dataset = build_dataset(str(val_path), tokenizer)

    lora_config = LoraConfig(**lora_cfg_dict)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        bf16=cfg.get("bf16", True) and torch.cuda.is_available(),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to=cfg.get("report_to", "none"),
        max_seq_length=cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDone. LoRA adapter + tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
