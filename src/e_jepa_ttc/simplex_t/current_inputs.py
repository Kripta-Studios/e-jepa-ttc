"""Verified current-only inputs; not an authorization to fit or invent histories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .cache import Normalizer, fit_normalizer, training_mass


def load_current_inputs(
    root: Path, fold: int, role: str, *, ancestry_sha256: str, allowed_sequences: set[str]
) -> dict[str, Any]:
    """Validate table bytes and per-row frozen producer lineage before array use."""
    if fold not in {0, 1, 2} or role not in {"inner_oof", "outer_dev"}:
        raise ValueError("unregistered current table")
    ancestry_path = root / "NESTED_ANCESTRY_AUDIT.json"
    if sha256(ancestry_path) != ancestry_sha256:
        raise ValueError("ancestry changed")
    ancestry = json.loads(ancestry_path.read_text())
    stem = root / "tables" / f"outer{fold}_{role}"
    index = json.loads((root / "FROZEN_EXPERT_TABLE_INDEX.json").read_text())
    entries = [row for row in index if row["outer_fold"] == fold and row["role"] == role]
    if len(entries) != 1 or entries[0]["ancestry_sha256"] != ancestry_sha256:
        raise ValueError("ambiguous or foreign table lineage")
    expected = entries[0]
    for suffix, key in ((".csv", "metadata_sha256"), (".npz", "arrays_sha256")):
        if sha256(stem.with_suffix(suffix)) != expected[key]:
            raise ValueError("table payload changed")
    columns = {
        "sample_token",
        "sequence_id",
        "target_ttc",
        "producer_outer_fold",
        "checkpoint_sha256",
        "pair_checkpoint_sha256",
    }
    if role == "inner_oof":
        columns.add("inner_fold")
    metadata = pd.read_csv(stem.with_suffix(".csv"), usecols=lambda name: name in columns)
    if set(metadata.columns) != columns or not metadata.sample_token.is_unique:
        raise ValueError("missing identity fields or duplicate query")
    if not set(metadata.sequence_id) <= allowed_sequences:
        raise ValueError("current input includes closed role")
    producers = {(p["outer_fold"], p["expert"], p["role"]): p for p in ancestry["producers"]}
    for row in metadata.to_dict("records"):
        family = f"inner{int(row['inner_fold'])}" if role == "inner_oof" else "outer_dev"
        if row["producer_outer_fold"] != fold:
            raise ValueError("wrong outer producer")
        a5 = producers[(fold, "A5", family)]
        pair = producers[(fold, "PAIR", family)]
        if pair["nested_a5_ancestor_sha256"] != a5["checkpoint_sha256"]:
            raise ValueError("PAIR teacher family mismatch")
        if row["pair_checkpoint_sha256"] != pair["checkpoint_sha256"]:
            raise ValueError("row uses another PAIR producer")
        for expert in ("A5", "C2F"):
            split = producers[(fold, expert, family)]["split_validation"]
            if (
                row["sequence_id"] in split["train_sequence_ids"]
                or row["sequence_id"] not in split["dev_sequence_ids"]
                or split["ancestry_validated"] is not True
            ):
                raise ValueError("row not excluded from expert training ancestors")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in ("features17", "expert_phase", "expert_ttc", "target_phase")
        }
    for name, value in arrays.items():
        if list(value.shape) != expected["arrays"][name]["shape"] or not np.isfinite(value).all():
            raise ValueError("current array schema/finite failure")
    if not np.array_equal(
        arrays["features17"][:, 8:11].astype(np.float32),
        arrays["expert_phase"].astype(np.float32),
    ):
        raise ValueError("RISK17 expert anchor columns disagree")
    return {
        "metadata": metadata,
        "arrays": arrays,
        "reference": expected,
        "role": role,
        "scientific_ready": False,
    }


def normalize_current_train(table: dict[str, Any]) -> tuple[Normalizer, np.ndarray]:
    """Use unique TRAIN rows and recomputed registered weighting, never dev."""
    if table["role"] != "inner_oof":
        raise ValueError("normalizer and training weights require inner_oof TRAIN")
    features = table["arrays"]["features17"]
    ids = np.arange(len(features), dtype=np.int64)
    normalizer = fit_normalizer(features, ids, np.ones(len(features), dtype=bool))
    metadata = table["metadata"]
    mass = training_mass(metadata.target_ttc.to_numpy(), metadata.sequence_id.to_numpy())
    return normalizer, mass
