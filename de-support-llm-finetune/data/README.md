# Dataset

## Schema

Each example in `data/processed/{train,val,test}.jsonl` is a JSON object:

```json
{
  "id": "0001",
  "system_prompt": "Du bist ein Assistent für den deutschen Kundenservice ...",
  "customer_email": "Sehr geehrtes Team, ich warte seit 10 Tagen auf mein Paket ...",
  "label": {
    "category": "Versand",
    "priority": "hoch",
    "sentiment": "negativ",
    "response_draft": "Sehr geehrte(r) Frau/Herr Müller, vielen Dank für Ihre Nachricht ..."
  }
}
```

- `category` ∈ {`Rechnung`, `Versand`, `Rückgabe`, `Technischer Support`, `Konto`}
- `priority` ∈ {`niedrig`, `mittel`, `hoch`}
- `sentiment` ∈ {`positiv`, `neutral`, `negativ`}
- `response_draft` — a short, polite German reply draft (2–4 sentences)

`src/prepare_dataset.py` turns each example into a chat-formatted training sample where
the model must output `json.dumps(label)` given `system_prompt` + `customer_email`.

## Generating the data

```bash
python data/generate_synthetic_data.py --n_examples 600 --seed 42 --out_dir data/processed
```

This produces `train.jsonl` (80%), `val.jsonl` (10%), `test.jsonl` (10%), fully
programmatically — no external download required, so the whole pipeline works offline.

## Using real data instead

For a stronger portfolio piece (or production use), replace the generator's output with
real, **anonymized** historical support tickets in the same JSONL schema. Remove any PII
(names, addresses, order/account numbers) before committing data to a public repository.
`data/processed/` is git-ignored by default for exactly this reason — regenerate or supply
your own data locally rather than committing it.
