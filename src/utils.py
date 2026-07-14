"""Shared project utilities."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
