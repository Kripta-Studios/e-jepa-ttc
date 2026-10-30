"""Verify immutable feature data and align sensor metadata by exact query identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from operational.evttc_transfer_audit.common import file_sha


def read(path: Path) -> dict[str, Any]:
    """Read a local UTF-8 JSON artifact."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_data(root: Path) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    """Validate completed cache bytes before any fitting or evaluation."""
    receipt = read(root / "COMPLETE.json")
    for filename, key in (
        ("training.npz", "training_npz_sha256"),
        ("POPULATION.json", "population_sha256"),
        ("TARGET_LEDGER.json", "targets_sha256"),
        ("BINDING.json", "binding_sha256"),
    ):
        if file_sha(root / filename) != receipt[key]:
            raise ValueError(f"Changed campaign input: {filename}")
    with np.load(root / "training.npz", allow_pickle=False) as source:
        values = {k: source[k] for k in source.files}
    rows = {"evttc:" + r["query_id"]: r for r in read(root / "POPULATION.json")["rows"]}
    if len(values["features"]) != receipt["rows"]:
        raise ValueError("Population count differs")
    return values, rows, receipt
