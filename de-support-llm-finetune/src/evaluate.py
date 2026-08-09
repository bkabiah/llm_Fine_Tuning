"""
Evaluates a model (base or LoRA-adapted) on the held-out test set:

  - JSON validity rate (does the output parse as JSON with the expected keys?)
  - Category / priority exact-match accuracy
  - ROUGE-L of response_draft vs. reference

Can be run twice — once with `--adapter_dir` omitted (zero-shot baseline) and
once with it set (fine-tuned model) — to reproduce the comparison table in the
README.

Example:
    python src/evaluate.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_dir outputs/qwen2.5-1.5b-support-lora \
        --test_file data/processed/test.jsonl \
        --report_path outputs/eval_report_finetuned.json

    python src/evaluate.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --test_file data/processed/test.jsonl \
        --report_path outputs/eval_report_baseline.json
"""
import argparse
import json
import sys
from pathlib import Path

import evaluate as hf_evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import build_prompt_only, load_jsonl  # noqa: E402

EXPECTED_KEYS = {"category", "priority", "sentiment", "response_draft"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", default=None,
                         help="Path to a trained LoRA adapter. Omit to evaluate the raw base model.")
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--report_path", default="outputs/eval_report.json")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optionally evaluate only the first N examples (fast iteration).")
    return parser.parse_args()


def try_parse_json(text: str):
    """Best-effort extraction of a JSON object from raw model output."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def load_model(base_model: str, adapter_dir: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    rows = load_jsonl(args.test_file)
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    rouge = hf_evaluate.load("rouge")

    n_valid_json = 0
    n_category_correct = 0
    n_priority_correct = 0
    predictions, references = [], []
    examples_out = []

    for row in rows:
        prompt = build_prompt_only(row, tokenizer)
        raw_output = generate(model, tokenizer, prompt, args.max_new_tokens)
        parsed = try_parse_json(raw_output)

        is_valid = bool(parsed) and EXPECTED_KEYS.issubset(parsed.keys())
        if is_valid:
            n_valid_json += 1
            if parsed.get("category") == row["label"]["category"]:
                n_category_correct += 1
            if parsed.get("priority") == row["label"]["priority"]:
                n_priority_correct += 1
            predictions.append(parsed.get("response_draft", ""))
        else:
            predictions.append("")
        references.append(row["label"]["response_draft"])

        examples_out.append({
            "id": row["id"],
            "raw_output": raw_output,
            "parsed": parsed,
            "reference": row["label"],
        })

    n = len(rows)
    rouge_scores = rouge.compute(predictions=predictions, references=references) if n else {}

    report = {
        "model": args.base_model,
        "adapter": args.adapter_dir,
        "n_examples": n,
        "json_validity_rate": n_valid_json / n if n else 0.0,
        "category_accuracy": n_category_correct / n if n else 0.0,
        "priority_accuracy": n_priority_correct / n if n else 0.0,
        "rouge": rouge_scores,
        "examples": examples_out[:10],  # keep report file small
    }

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))
    print(f"\nFull report (incl. sample generations) written to {args.report_path}")


if __name__ == "__main__":
    main()
