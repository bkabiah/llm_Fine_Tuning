"""Shared helpers for loading YAML configs used across the training/eval scripts."""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    """Resolve a path relative to the project root (parent of src/)."""
    p = Path(path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parent.parent
    return project_root / p
