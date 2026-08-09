"""
Minimal Gradio web demo for the fine-tuned German customer-support model.

    python app/demo_gradio.py --model_dir outputs/merged-model
"""
import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import SYSTEM_PROMPT, respond  # noqa: E402

EXAMPLE_EMAILS = [
    "Hallo, mein Name ist Anna Schmidt. Ich warte seit 12 Tagen auf mein Paket "
    "(Bestellnummer DE-482913) und bin sehr verärgert. Wo bleibt meine Bestellung?",
    "Sehr geehrtes Team, könnten Sie mir bitte eine Rechnungskopie für Bestellung "
    "DE-118823 zusenden? Ich benötige diese für meine Unterlagen.",
    "Hallo, meine Smartwatch verbindet sich nicht per Bluetooth, obwohl ich die "
    "Anleitung befolgt habe. Können Sie mir helfen?",
]


def build_app(model, tokenizer, max_new_tokens: int):
    def handle(email: str):
        if not email.strip():
            return "Bitte geben Sie eine Kunden-E-Mail ein."
        raw = respond(model, tokenizer, email, max_new_tokens)
        try:
            parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, ValueError):
            return raw

    with gr.Blocks(title="Deutscher Kundenservice-Assistent") as demo:
        gr.Markdown(
            "# 🇩🇪 Deutscher Kundenservice-Assistent\n"
            "LoRA-feinabgestimmtes LLM zur Klassifikation & Antwortentwurf-Generierung "
            "für Kundenservice-E-Mails."
        )
        with gr.Row():
            inp = gr.Textbox(label="Kunden-E-Mail", lines=6, placeholder="E-Mail hier einfügen ...")
            out = gr.Code(label="Modellausgabe (JSON)", language="json")
        btn = gr.Button("Analysieren", variant="primary")
        btn.click(fn=handle, inputs=inp, outputs=out)
        gr.Examples(examples=EXAMPLE_EMAILS, inputs=inp)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

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

    demo = build_app(model, tokenizer, args.max_new_tokens)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
