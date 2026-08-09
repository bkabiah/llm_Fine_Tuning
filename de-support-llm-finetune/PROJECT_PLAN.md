# Projektplan: German Customer-Support LLM Fine-Tuning

## 1. Ziel

Ein kleines, offen lizenziertes LLM (Qwen2.5-1.5B-Instruct) mittels LoRA/QLoRA so
feinabstimmen, dass es deutsche Kundenservice-E-Mails automatisch klassifiziert
(Kategorie, Priorität, Sentiment) und einen Antwortentwurf im JSON-Format erzeugt.

## 2. Problemstellung

Support-Teams verbringen viel Zeit mit dem manuellen Sichten und Beantworten
eingehender E-Mails. Ein Modell, das strukturierte Vorschläge liefert, spart Zeit und
sorgt für konsistente Antwortqualität. Das Projekt ist bewusst klein genug gehalten,
um auf einer einzelnen Consumer-GPU (oder Google Colab, kostenlos) trainierbar zu sein.

## 3. Vorgehen (Milestones)

| # | Milestone | Beschreibung | Artefakt |
|---|---|---|---|
| 1 | Datenschema definieren | Input/Output-Format, Kategorien, Prioritätsstufen festlegen | `data/README.md` |
| 2 | Synthetische Daten generieren | Template-basierter Generator, Train/Val/Test-Split | `data/generate_synthetic_data.py` |
| 3 | Dataset-Aufbereitung | JSONL → HF `Dataset`, Chat-Template, Tokenisierung | `src/prepare_dataset.py` |
| 4 | Baseline messen | Zero-Shot-Performance des Basismodells ohne Fine-Tuning | `src/evaluate.py` |
| 5 | LoRA-Training | SFTTrainer mit LoRA/QLoRA, Hyperparameter-Konfiguration | `src/train.py`, `configs/*.yaml` |
| 6 | Evaluierung | JSON-Validität, Klassifikationsgenauigkeit, ROUGE-L, Vergleich zur Baseline | `outputs/eval_report.json` |
| 7 | Merge & Export | LoRA-Adapter in Basismodell mergen, für Deployment exportieren | `src/merge_and_export.py` |
| 8 | Demo | CLI-Inferenz + Gradio-Weboberfläche | `src/inference.py`, `app/demo_gradio.py` |
| 9 | Tests & CI | Schema-Tests, GitHub-Actions-Workflow | `tests/`, `.github/workflows/ci.yml` |
| 10 | Dokumentation | README, Ergebnisse, Limitationen | `README.md` |

## 4. Modellwahl

- **Basismodell:** `Qwen/Qwen2.5-1.5B-Instruct` — klein genug für Consumer-Hardware,
  gute mehrsprachige/deutsche Fähigkeiten, permissive Lizenz, aktives Ökosystem.
- **Alternative für sehr limitierte Hardware:** `Qwen/Qwen2.5-0.5B-Instruct`.
- **Methode:** LoRA (Rang 16, Alpha 32, Ziel-Layer: `q_proj`, `k_proj`, `v_proj`,
  `o_proj`); optional QLoRA (4-bit NF4-Quantisierung) für GPUs mit wenig VRAM.

## 5. Evaluierungsstrategie

1. **JSON-Validität** — Anteil der Modellantworten, die als valides JSON mit den
   erwarteten Feldern (`category`, `priority`, `sentiment`, `response_draft`)
   geparst werden können.
2. **Klassifikationsgenauigkeit** — Exact-Match auf `category` und `priority`
   gegenüber Referenzlabels.
3. **Textqualität der Antwort** — ROUGE-L zwischen generiertem `response_draft`
   und Referenzantwort.
4. **Vergleich Baseline vs. Fine-Tuned** — dieselbe Metrik-Pipeline läuft für das
   unveränderte Basismodell (Zero-Shot mit Prompt-Instruktionen) und für das
   fein­abgestimmte Modell.

## 6. Risiken & Gegenmaßnahmen

| Risiko | Auswirkung | Gegenmaßnahme |
|---|---|---|
| Synthetische Daten zu repetitiv/unrealistisch | Modell generalisiert schlecht | Große Template- und Slot-Variationsbreite; klar als Limitation dokumentiert |
| Kein Zugriff auf starke GPU | Training zu langsam/nicht möglich | QLoRA (4-bit) + kleines Modell + Colab-Anleitung |
| Halluzinierte / falsch formatierte JSON-Ausgaben | Unbrauchbar in Produktion | Strenger System-Prompt, JSON-Validitäts-Metrik, Few-Shot-Beispiele im Prompt |
| Überanpassung (Overfitting) bei kleinem Datensatz | Schlechte Generalisierung | Val-Split zur frühzeitigen Erkennung, moderate Epochenzahl (2–3), LoRA statt Full-FT |

## 7. Zeitplan (Beispiel, für ein Wochenend-/Abendprojekt)

| Tag | Aufgabe |
|---|---|
| 1 | Datenschema + Generator fertigstellen, Beispiele manuell prüfen |
| 2 | Trainings-Pipeline aufsetzen, ersten Trainingslauf starten |
| 3 | Evaluierung implementieren, Ergebnisse mit Baseline vergleichen |
| 4 | Merge/Export, Gradio-Demo, README & Ergebnisse finalisieren |

## 8. Erweiterungsideen (optional, für "Nice-to-have" im Interview)

- Ersetzen der synthetischen Daten durch echte (anonymisierte) Support-Tickets
- DPO/RLHF-Feinschliff für Ton und Markenkonsistenz
- Mehrsprachige Erweiterung (DE/EN gemischt)
- Deployment als kleine FastAPI + Docker-Service
