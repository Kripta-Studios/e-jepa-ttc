"""Regenerable table helpers; values always come from supplied JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metric_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load JSON objects without synthesizing or filling metric values."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Metric artifact {source} must contain a JSON object.")
        rows.append(payload)
    return rows


def write_json_table(paths: list[str | Path], output: str | Path) -> None:
    """Write an auditable JSON table with source paths and no derived claims."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_paths": [Path(path).as_posix() for path in paths],
        "rows": load_metric_rows(paths),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["load_metric_rows", "write_json_table"]
