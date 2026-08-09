"""
Merges a trained LoRA adapter into the base model weights and saves the
result as a standalone model directory, ready for deployment or further
quantization (e.g. GGUF conversion with llama.cpp).

Example:
    python src/merge_and_export.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_dir outputs/qwen2.5-1.5b-support-lora \
        --output_dir outputs/merged-model
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/merged-model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter into base model weights ...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. You can now load this directory directly with AutoModelForCausalLM, "
          "or convert it (e.g. to GGUF with llama.cpp) for lightweight local inference.")


if __name__ == "__main__":
    main()
