"""Strict population admission and isolated artifacts for decision recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT.parent / "e-jepa-ttc-v18-joint-measurement"
SOURCE_RUN = SOURCE_ROOT / "artifacts/joint_measurement_20261012"
RUN = ROOT / "artifacts/decision_recovery_20261017"
EXPERTS = {
    "A5": ("", "A5-P2-FROZEN"),
    "NATIVE": ("", "NATIVE-P2-FT-S7"),
    "CORR": ("", "PAIR-CORR-S7"),
    "C1": ("adaptive/C1_geometry", "PAIR-CORR-GEOMETRY-S7"),
    "B1": ("adaptive/B1_correspondence", "B1-CORRESPONDENCE-STAGE2-S7"),
    "FULL": ("adaptive/A1_adaptation_depth", "A1-FULL-S7"),
    "LAST": ("adaptive/A1_adaptation_depth", "A1-LAST-BLOCK-S7"),
}
IDENTITY = ["sample_token", "group_id", "sequence_id", "target_ttc", "role"]


def sha(path: Path) -> str:
    """Hash actual bytes rather than trusting a producer's file name."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write only inside the new campaign, with no nonstandard NaN JSON."""
    path = Path(path).resolve()
    if not path.is_relative_to(RUN.resolve()):
        raise ValueError("V19 writes must stay inside its own artifact directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prediction_path(role: str, name: str) -> Path:
    if role not in {"P", "H"} or name not in EXPERTS:
        raise ValueError("Only the registered P/H experts may be opened in this stage")
    folder, fit = EXPERTS[name]
    return SOURCE_RUN / folder / "predictions" / role / fit / "predictions.parquet"


def source_bindings(role: str, names: list[str]) -> dict[str, str]:
    bindings = {}
    for name in names:
        path = prediction_path(role, name)
        for file in (path, path.parent / "COMPLETE.json", path.parent / "METRICS.json"):
            bindings[str(file.relative_to(SOURCE_ROOT)).replace("\\", "/")] = sha(file)
    return bindings


def load_population(role: str, names: list[str]) -> dict[str, pd.DataFrame]:
    """Reject omissions, reordering, changed targets, and changed prediction bytes."""
    if not names or len(set(names)) != len(names):
        raise ValueError("A nonempty unique expert list is required")
    result: dict[str, pd.DataFrame] = {}
    reference = None
    for name in names:
        path = prediction_path(role, name)
        receipt = read_json(path.parent / "COMPLETE.json")
        if receipt["status"] != "COMPLETE" or receipt["sha256"] != sha(path):
            raise ValueError(f"Unverified prediction bytes: {role}/{name}")
        table = pd.read_parquet(path)
        required = {*IDENTITY, "prediction_ttc", "raw_phase"}
        if not required.issubset(table.columns):
            raise ValueError(f"Missing prediction fields: {role}/{name}")
        if (
            len(table) != receipt["rows"]
            or len(table) != {"P": 53890, "H": 18422}[role]
            or table.sample_token.isna().any()
            or table.sample_token.duplicated().any()
            or table[IDENTITY].isna().any().any()
            or set(table.role) != {role}
        ):
            raise ValueError(f"Incomplete or duplicate cohort: {role}/{name}")
        values = table[["target_ttc", "prediction_ttc"]].to_numpy(dtype=np.float64)
        if not (np.isfinite(values).all() and ((values < 0) | (values > 0.1)).all()):
            raise ValueError(f"Invalid native outputs must be audited, not filtered: {name}")
        identity = table[IDENTITY].reset_index(drop=True)
        if reference is not None and not identity.equals(reference):
            raise ValueError(f"Population, exact target bytes or order changed: {role}/{name}")
        reference = identity
        result[name] = table
    return result


def score(table: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    """Use the unchanged full metric/gate implementation, preserving invalid outcomes."""
    from operational.range_transfer.analyze import metrics

    return metrics(
        table.target_ttc.to_numpy(dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
        table.group_id.to_numpy(),
    )
