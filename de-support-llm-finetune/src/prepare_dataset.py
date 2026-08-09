"""
Converts the JSONL files produced by data/generate_synthetic_data.py into a
Hugging Face `Dataset` of chat-formatted training examples ready for TRL's
SFTTrainer.

Each row becomes a 3-turn chat:
  system: instructions (from the example's system_prompt)
  user:   the raw customer email
  assistant: json.dumps(label), the target the model must learn to produce
"""
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_chat_messages(example: Dict) -> List[Dict[str, str]]:
    """Build the 3-message chat list for one raw example."""
    target = json.dumps(example["label"], ensure_ascii=False)
    return [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["customer_email"]},
        {"role": "assistant", "content": target},
    ]


def build_dataset(jsonl_path: str, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Load a JSONL file and return a Dataset with a single 'text' column
    containing the fully rendered chat (system+user+assistant), suitable for
    TRL's SFTTrainer (which trains on the whole formatted string)."""
    rows = load_jsonl(jsonl_path)
    texts = []
    for row in rows:
        messages = to_chat_messages(row)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def build_prompt_only(example: Dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Build just the system+user portion (with generation prompt) for inference
    / evaluation, i.e. what the model sees before it must generate the JSON
    label."""
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["customer_email"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


if __name__ == "__main__":
    # Quick sanity check / demo when run directly.
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/processed/train.jsonl")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        raise SystemExit(
            f"{args.jsonl} not found. Run data/generate_synthetic_data.py first."
        )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = build_dataset(args.jsonl, tok)
    print(ds)
    print("---- Example 0 ----")
    print(ds[0]["text"])
