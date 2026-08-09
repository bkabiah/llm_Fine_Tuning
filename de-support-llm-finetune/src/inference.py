"""
Small interactive CLI to try out the (merged or adapter-based) model on your
own German customer emails.

Examples:
    # merged model
    python src/inference.py --model_dir outputs/merged-model

    # base model + separate LoRA adapter (not merged)
    python src/inference.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_dir outputs/qwen2.5-1.5b-support-lora
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "Du bist ein Assistent fuer den deutschen Kundenservice eines Onlineshops. "
    "Lies die Kunden-E-Mail und antworte AUSSCHLIESSLICH mit einem JSON-Objekt mit "
    "den Feldern 'category' (Rechnung, Versand, Rueckgabe, Technischer Support, Konto), "
    "'priority' (niedrig, mittel, hoch), 'sentiment' (positiv, neutral, negativ) und "
    "'response_draft' (ein hoeflicher, kurzer Antwortentwurf auf Deutsch)."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=None, help="Path to a merged model directory.")
    parser.add_argument("--base_model", default=None, help="Base model name (used with --adapter_dir).")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter directory.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    return parser.parse_args()


def load(args):
    if args.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    elif args.base_model and args.adapter_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        raise SystemExit("Provide either --model_dir, or both --base_model and --adapter_dir.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@torch.no_grad()
def respond(model, tokenizer, email: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    model, tokenizer = load(args)

    print("German customer-support triage assistant. Type a customer email, or 'exit' to quit.\n")
    while True:
        try:
            email = input(">> Kunden-E-Mail: ")
        except (EOFError, KeyboardInterrupt):
            break
        if email.strip().lower() in {"exit", "quit"}:
            break
        if not email.strip():
            continue

        raw = respond(model, tokenizer, email, args.max_new_tokens)
        print("\n--- Modellausgabe ---")
        try:
            parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        except (json.JSONDecodeError, ValueError):
            print(raw)
        print()


if __name__ == "__main__":
    main()
