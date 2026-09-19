# 🇩🇪 German Customer-Support LLM Fine-Tuning

**Parameter-efficient fine-tuning (LoRA/QLoRA) of an open-weight LLM to triage and draft
replies to German customer-support emails — structured JSON output, end-to-end pipeline,
reproducible on a single consumer GPU (or free Colab).**

---

## Einführung
Dieses Projekt demonstriert die parameter-effiziente Feinabstimmung (LoRA/QLoRA) eines kompakten Open-Source-Sprachmodells (Qwen2.5-1.5B-Instruct) speziell für den deutschen Kundenservice. Ziel ist es, eingehende Support-E-Mails automatisch zu analysieren, nach Kategorie, Priorität und Sentiment zu klassifizieren und einen professionellen Antwortentwurf im strikten JSON-Format zu generieren. Die vollständig reproduzierbare End-to-End-Pipeline umfasst synthetische Datengenerierung, Training, automatisierte Evaluierung und eine interaktive Gradio-Demo. Dadurch eignet sich das Projekt ideal als Portfolio-Stück für AI-Engineering-Rollen, da es moderne LLM-Techniken mit einem greifbaren Business-Use-Case verbindet – und das ressourcenschonend auf handelsüblicher Hardware.

---

> 🇩🇪 **Kurzfassung (Deutsch):** Dieses Projekt zeigt, wie ein kleines Open-Source-LLM
> (Qwen2.5-1.5B-Instruct) mittels LoRA/QLoRA auf deutsche Kundenservice-E-Mails feinabgestimmt
> wird. Das Modell klassifiziert Anfragen (Kategorie, Priorität, Sentiment) und formuliert
> automatisch eine passende Antwort im JSON-Format. Enthalten sind: Datengenerierung,
> Trainings-Pipeline, Evaluierung, Inferenz-CLI und eine Gradio-Demo — vollständig
> dokumentiert und reproduzierbar.

---


```mermaid

graph TD
    A[📧 Rohe Kunden-E-Mail<br/>Deutsch] --> B(⚙️ Prompt Engineering &<br/>Chat-Template Anwendung)
    B --> C{🧠 Qwen2.5-1.5B-Instruct<br/>+ LoRA Adapter (r=16)}
    C -->|Generierung| D[📦 Strukturierte JSON-Ausgabe]
    
    D --> E1[🏷️ Kategorie<br/>z.B. Versand, Rechnung]
    D --> E2[🔥 Priorität<br/>niedrig, mittel, hoch]
    D --> E3[😊 Sentiment<br/>positiv, neutral, negativ]
    D --> E4[✍️ Antwortentwurf<br/>Höflicher, kurzer Text]

    style C fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style D fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style A fill:#fff3e0,stroke:#e65100,stroke-width:2px

```

---


## 1. Problem Statement & Business Value

Support teams lose time triaging and drafting first-response emails. This project fine-tunes
a small instruction-tuned LLM so that, given a raw German customer email, it outputs:

```json
{
  "category": "Versand",
  "priority": "hoch",
  "sentiment": "negativ",
  "response_draft": "Sehr geehrte(r) Frau/Herr ..., vielen Dank für Ihre Nachricht ..."
}
```

This mirrors a realistic, demo-able business use case (ticket triage + draft reply
assistant) that is easy to explain to a non-technical interviewer while still exercising
the full modern LLM fine-tuning stack.

## 2. What This Project Demonstrates

- Synthetic dataset construction & schema design for instruction tuning
- Parameter-efficient fine-tuning with **LoRA** and optional **QLoRA (4-bit)** via
  `transformers`, `peft`, `trl`, `bitsandbytes`
- Structured-output ("JSON-mode") instruction tuning and evaluation
- Automated evaluation (JSON validity rate, category/priority accuracy, ROUGE-L) vs. a
  zero-shot baseline
- Adapter merging & export for deployment
- A minimal Gradio demo for showcasing the model
- Basic tests + CI (GitHub Actions)

## 3. Architecture

```
Raw customer email (German)
        │
        ▼
 ┌─────────────────────────┐
 │  Base model (frozen)    │  Qwen2.5-1.5B-Instruct
 │  + LoRA adapters (r=16) │  ← trained on synthetic support-ticket data
 └─────────────────────────┘
        │
        ▼
 Structured JSON: category, priority, sentiment, response_draft
```

## 4. Repository Structure

```
de-support-llm-finetune/
├── README.md
├── PROJECT_PLAN.md            # Planning doc: scope, milestones, risks
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── generate_synthetic_data.py   # builds train/val/test JSONL
│   └── README.md
├── configs/
│   ├── lora_config.yaml
│   └── training_config.yaml
├── src/
│   ├── config.py
│   ├── prepare_dataset.py     # JSONL -> HF Dataset with chat template
│   ├── train.py                # LoRA/QLoRA fine-tuning (SFTTrainer)
│   ├── evaluate.py             # metrics vs. baseline
│   ├── merge_and_export.py     # merge adapter into base model
│   └── inference.py            # CLI inference
├── app/
│   └── demo_gradio.py          # small web demo
├── tests/
│   └── test_data_format.py
└── .github/workflows/ci.yml
```

## 5. Setup

```bash
git clone https://github.com/<your-username>/de-support-llm-finetune.git
cd de-support-llm-finetune
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10/3.11. GPU with ≥8 GB VRAM recommended for LoRA on the 1.5B model;
enable `--load_in_4bit` in `configs/training_config.yaml` for QLoRA on smaller GPUs (≥6 GB)
or use a free Google Colab T4.

## 6. Usage

### 6.1 Generate the dataset

```bash
python data/generate_synthetic_data.py --n_examples 600 --seed 42
# -> data/processed/{train,val,test}.jsonl
```

The generator programmatically composes realistic German support emails from templates,
customer names, order numbers and issue variations across 5 categories (Rechnung, Versand,
Rückgabe, Technischer Support, Konto), so the pipeline runs fully offline with no external
dataset dependency. See `data/README.md` for the schema and notes on swapping in a real
dataset (e.g. your own historical support tickets, anonymized).

### 6.2 Fine-tune with LoRA

```bash
python src/train.py \
  --config configs/training_config.yaml \
  --lora_config configs/lora_config.yaml \
  --output_dir outputs/qwen2.5-1.5b-support-lora
```

Key flags (also settable in the YAML configs):

| Flag | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | base model |
| `--load_in_4bit` | `false` | enable QLoRA (bitsandbytes 4-bit) |
| `--num_train_epochs` | `3` | training epochs |
| `--learning_rate` | `2e-4` | LoRA learning rate |
| `--lora_r` / `--lora_alpha` | `16` / `32` | LoRA rank / scaling |

### 6.3 Evaluate

```bash
python src/evaluate.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen2.5-1.5b-support-lora \
  --test_file data/processed/test.jsonl \
  --report_path outputs/eval_report.json
```

Reports, per model (fine-tuned vs. zero-shot base model):

- **JSON validity rate** — % of generations that parse as valid JSON with the expected keys
- **Category accuracy** / **priority accuracy** — exact-match on the classification fields
- **ROUGE-L** — `response_draft` vs. reference reply

### 6.4 Merge & export

```bash
python src/merge_and_export.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen2.5-1.5b-support-lora \
  --output_dir outputs/merged-model
```

### 6.5 Try it out

```bash
python src/inference.py --model_dir outputs/merged-model
# or, without merging:
python src/inference.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_dir outputs/qwen2.5-1.5b-support-lora
```

### 6.6 Gradio demo

```bash
python app/demo_gradio.py --model_dir outputs/merged-model
```

## 7. Results

*Fill this in after running training on your machine/Colab — numbers depend on GPU,
epochs, and dataset size. Template:*

| Model | JSON validity | Category acc. | Priority acc. | ROUGE-L |
|---|---|---|---|---|
| Base (zero-shot) | … | … | … | … |
| Fine-tuned (LoRA) | … | … | … | … |

`src/evaluate.py` writes exactly this table's data to `outputs/eval_report.json`.

## 8. Limitations & Future Work

- The dataset is synthetically generated (template-based); for production use, replace with
  real, anonymized support tickets and human-reviewed reference replies.
- No RLHF/DPO step — could be added to better align tone/style with brand voice.
- Only single-turn tickets are modeled; multi-turn conversation history is a natural
  extension (see `src/prepare_dataset.py` for where to add conversation context).
- Evaluation is automatic (ROUGE, accuracy); a small human-eval pass on generated replies
  is recommended before any real deployment.

## 9. License

Code released under the [MIT License](LICENSE). Base model licenses (e.g. Qwen) apply
separately — check the respective model card before commercial use.

## 10. About

Portfolio project by [Your Name] — built to demonstrate applied LLM fine-tuning skills
for AI Engineer roles in Germany. Feedback and PRs welcome.

[LinkedIn](#) · [Portfolio](#) · [Email](#)
