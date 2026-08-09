"""
Generates a synthetic dataset of German customer-support emails with structured
labels (category, priority, sentiment, response_draft) for instruction fine-tuning.

Fully offline / template-based on purpose, so the whole project pipeline is
reproducible without needing an external dataset download. See data/README.md
for the schema and notes on swapping in real data.

Usage:
    python data/generate_synthetic_data.py --n_examples 600 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "Du bist ein Assistent fuer den deutschen Kundenservice eines Onlineshops. "
    "Lies die Kunden-E-Mail und antworte AUSSCHLIESSLICH mit einem JSON-Objekt mit "
    "den Feldern 'category' (Rechnung, Versand, Rueckgabe, Technischer Support, Konto), "
    "'priority' (niedrig, mittel, hoch), 'sentiment' (positiv, neutral, negativ) und "
    "'response_draft' (ein hoeflicher, kurzer Antwortentwurf auf Deutsch)."
)

FIRST_NAMES = ["Anna", "Max", "Julia", "Tobias", "Sabine", "Michael", "Laura",
               "Stefan", "Nicole", "Daniel", "Katharina", "Jonas", "Melanie",
               "Peter", "Sandra", "Kevin", "Petra", "Markus"]
LAST_NAMES = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
              "Wagner", "Becker", "Hoffmann", "Schulz", "Koch", "Richter",
              "Klein", "Wolf", "Neumann", "Schwarz"]
PRODUCTS = ["Laufschuhe", "Kaffeemaschine", "Bluetooth-Kopfhoerer", "Laptop-Tasche",
            "Winterjacke", "Wasserkocher", "Smartwatch", "Buerostuhl",
            "Staubsauger", "Rucksack", "Fahrradhelm", "Tablet-Huelle"]

CATEGORIES = ["Rechnung", "Versand", "Rückgabe", "Technischer Support", "Konto"]
PRIORITIES = ["niedrig", "mittel", "hoch"]
SENTIMENTS = ["positiv", "neutral", "negativ"]


def order_number(rng: random.Random) -> str:
    return f"DE-{rng.randint(100000, 999999)}"


def greeting(rng: random.Random, name: str) -> str:
    return "Sehr geehrtes Team,\n\n" if rng.random() < 0.3 else f"Hallo,\n\nmein Name ist {name}. "


def sign_off(name: str) -> str:
    return f"\n\nMit freundlichen Gruessen\n{name}"


# --- Template bank: (category, sentiment, priority, email_fn, response_fn) ---

def build_versand(rng, name, product, order):
    days_waiting = rng.randint(3, 21)
    negative = days_waiting > 7
    email = (
        f"{greeting(rng, name)}ich habe am {rng.randint(1, 27)}.{rng.randint(1, 12)}. die "
        f"{product} bestellt (Bestellnummer {order}), aber das Paket ist seit "
        f"{days_waiting} Tagen nicht angekommen. Koennen Sie mir sagen, wo sich meine "
        f"Sendung befindet?{sign_off(name)}"
    )
    priority = "hoch" if days_waiting > 10 else ("mittel" if negative else "niedrig")
    sentiment = "negativ" if negative else "neutral"
    response = (
        f"Sehr geehrte(r) {name}, vielen Dank fuer Ihre Nachricht und Entschuldigung fuer "
        f"die Verzoegerung bei Bestellung {order}. Wir pruefen den Sendungsstatus umgehend "
        f"und melden uns innerhalb von 24 Stunden mit einem Update bei Ihnen."
    )
    return email, priority, sentiment, response


def build_rueckgabe(rng, name, product, order):
    reason = rng.choice(["passt nicht", "ist beschaedigt angekommen", "entspricht nicht der Beschreibung"])
    email = (
        f"{greeting(rng, name)}ich moechte die {product} aus Bestellung {order} zurueckgeben, da sie "
        f"{reason}. Wie ist der Ablauf fuer eine Rueckgabe und wann bekomme ich mein Geld zurueck?"
        f"{sign_off(name)}"
    )
    priority = "mittel" if "beschaedigt" in reason else "niedrig"
    sentiment = "negativ" if "beschaedigt" in reason else "neutral"
    response = (
        f"Sehr geehrte(r) {name}, das tut uns leid zu hoeren. Bitte nutzen Sie das "
        f"Retourenlabel in Ihrem Kundenkonto fuer Bestellung {order}. Nach Erhalt der "
        f"Rueckgabe erstatten wir den Betrag innerhalb von 5 Werktagen."
    )
    return email, priority, sentiment, response


def build_rechnung(rng, name, product, order):
    amount = rng.choice([19.99, 49.90, 89.00, 129.50, 249.00, 15.00])
    duplicate = rng.random() < 0.5
    if duplicate:
        email = (
            f"{greeting(rng, name)}bei mir wurde fuer die Bestellung {order} ({product}) ein Betrag "
            f"von {amount:.2f} Euro zweimal abgebucht. Bitte pruefen Sie das und erstatten Sie "
            f"den doppelten Betrag.{sign_off(name)}"
        )
        priority, sentiment = "hoch", "negativ"
        response = (
            f"Sehr geehrte(r) {name}, vielen Dank fuer den Hinweis. Wir pruefen die Doppelbuchung "
            f"zu Bestellung {order} umgehend im Finanzsystem und erstatten den zu viel gezahlten "
            f"Betrag von {amount:.2f} Euro innerhalb von 3 Werktagen."
        )
    else:
        email = (
            f"{greeting(rng, name)}koennten Sie mir bitte eine Rechnungskopie fuer Bestellung {order} "
            f"({product}) zusenden? Ich benoetige diese fuer meine Unterlagen.{sign_off(name)}"
        )
        priority, sentiment = "niedrig", "neutral"
        response = (
            f"Sehr geehrte(r) {name}, gerne senden wir Ihnen die Rechnung zu Bestellung {order} "
            f"als PDF an Ihre hinterlegte E-Mail-Adresse. Sie erhalten diese in Kuerze."
        )
    return email, priority, sentiment, response


def build_technischer_support(rng, name, product):
    issue = rng.choice(["laesst sich nicht einschalten", "verbindet sich nicht per Bluetooth",
                         "zeigt eine Fehlermeldung", "funktioniert nur teilweise"])
    email = (
        f"{greeting(rng, name)}meine {product} {issue}, obwohl ich die Anleitung befolgt habe. "
        f"Koennen Sie mir helfen, das Problem zu loesen?{sign_off(name)}"
    )
    priority, sentiment = "mittel", "negativ"
    response = (
        f"Sehr geehrte(r) {name}, das tut uns leid. Bitte versuchen Sie zunaechst einen "
        f"Neustart des Geraets sowie ein Zuruecksetzen auf Werkseinstellungen. Sollte das "
        f"Problem bei Ihrer {product} weiterhin bestehen, senden wir Ihnen gerne kostenlos "
        f"ein Ersatzgeraet zu."
    )
    return email, priority, sentiment, response


def build_konto(rng, name):
    issue = rng.choice(["mein Passwort vergessen", "meine Adresse aktualisieren",
                         "mein Konto loeschen", "meine E-Mail-Adresse aendern"])
    email = (
        f"{greeting(rng, name)}ich moechte {issue}, finde aber die passende Option nicht in meinem "
        f"Kundenkonto. Koennen Sie mir bitte weiterhelfen?{sign_off(name)}"
    )
    priority = "hoch" if "loeschen" in issue or "Passwort" in issue else "niedrig"
    sentiment = "neutral"
    response = (
        f"Sehr geehrte(r) {name}, gerne helfen wir Ihnen weiter. Unter 'Mein Konto' > "
        f"'Einstellungen' koennen Sie die gewuenschte Aenderung vornehmen. Falls Sie nicht "
        f"weiterkommen, senden wir Ihnen einen direkten Link zur entsprechenden Seite zu."
    )
    return email, priority, sentiment, response


def build_positive_feedback(rng, name, product, order):
    email = (
        f"{greeting(rng, name)}ich wollte mich nur kurz bedanken - die {product} aus Bestellung "
        f"{order} ist super angekommen und die Lieferung war sehr schnell! Weiter so."
        f"{sign_off(name)}"
    )
    priority, sentiment = "niedrig", "positiv"
    response = (
        f"Sehr geehrte(r) {name}, vielen herzlichen Dank fuer Ihr tolles Feedback! Wir freuen "
        f"uns sehr, dass Sie mit Ihrer {product} und der Lieferung zufrieden sind."
    )
    return email, priority, sentiment, response


def generate_examples(n: int, rng: random.Random):
    examples = []
    builders = [
        ("Versand", build_versand),
        ("Rückgabe", build_rueckgabe),
        ("Rechnung", build_rechnung),
        ("Technischer Support", build_technischer_support),
        ("Konto", build_konto),
        ("Versand", build_positive_feedback),  # positive feedback ~ shipping-adjacent
    ]
    for i in range(n):
        category_name, builder = rng.choice(builders)
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        product = rng.choice(PRODUCTS)
        order = order_number(rng)

        if builder is build_konto:
            email, priority, sentiment, response = builder(rng, name)
        elif builder is build_technischer_support:
            email, priority, sentiment, response = builder(rng, name, product)
        else:
            email, priority, sentiment, response = builder(rng, name, product, order)

        examples.append({
            "id": f"{i:04d}",
            "system_prompt": SYSTEM_PROMPT,
            "customer_email": email,
            "label": {
                "category": category_name,
                "priority": priority,
                "sentiment": sentiment,
                "response_draft": response,
            },
        })
    return examples


def split_and_write(examples, out_dir: Path, rng: random.Random):
    rng.shuffle(examples)
    n = len(examples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows):4d} examples -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_examples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = generate_examples(args.n_examples, rng)
    split_and_write(examples, Path(args.out_dir), rng)


if __name__ == "__main__":
    main()
