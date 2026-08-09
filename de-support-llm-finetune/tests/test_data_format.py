"""
Sanity tests for the synthetic data generator and its output schema.
Run with: pytest tests/
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from generate_synthetic_data import (  # noqa: E402
    CATEGORIES,
    PRIORITIES,
    SENTIMENTS,
    generate_examples,
)

EXPECTED_LABEL_KEYS = {"category", "priority", "sentiment", "response_draft"}
EXPECTED_EXAMPLE_KEYS = {"id", "system_prompt", "customer_email", "label"}


def test_generate_examples_count():
    rng = random.Random(0)
    examples = generate_examples(20, rng)
    assert len(examples) == 20


def test_example_schema():
    rng = random.Random(1)
    examples = generate_examples(50, rng)
    for ex in examples:
        assert EXPECTED_EXAMPLE_KEYS.issubset(ex.keys())
        assert EXPECTED_LABEL_KEYS.issubset(ex["label"].keys())
        assert isinstance(ex["customer_email"], str) and len(ex["customer_email"]) > 10
        assert isinstance(ex["label"]["response_draft"], str) and len(ex["label"]["response_draft"]) > 10


def test_label_values_are_valid():
    rng = random.Random(2)
    examples = generate_examples(50, rng)
    for ex in examples:
        assert ex["label"]["category"] in CATEGORIES
        assert ex["label"]["priority"] in PRIORITIES
        assert ex["label"]["sentiment"] in SENTIMENTS


def test_examples_are_json_serializable():
    rng = random.Random(3)
    examples = generate_examples(10, rng)
    for ex in examples:
        # Should not raise
        json.dumps(ex, ensure_ascii=False)


def test_determinism_given_seed():
    examples_a = generate_examples(30, random.Random(123))
    examples_b = generate_examples(30, random.Random(123))
    assert examples_a == examples_b
