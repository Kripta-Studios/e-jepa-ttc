"""Trivial TTC baselines computed from train targets only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.data.targets import load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.utils.io import write_structured


def _targets_for_sequences(sequences: list[DatasetSequence], sequence_ids: list[str]) -> np.ndarray:
    by_id = {sequence.sequence_id: sequence for sequence in sequences}
    targets: list[np.ndarray] = []
    for sequence_id in sequence_ids:
        if sequence_id not in by_id:
            msg = f"Unknown sequence in split: {sequence_id}"
            raise ValueError(msg)
        ttc_csv = by_id[sequence_id].resolve("ttc_csv")
        if ttc_csv is None:
            msg = f"Sequence {sequence_id} does not define ttc_csv."
            raise ValueError(msg)
        targets.append(load_ttc_csv(ttc_csv)["ttc_s"])
    if not targets:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(targets).astype(np.float64)


def run_trivial_baseline(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute mean and median train-set TTC baselines across all splits."""

    sequences = read_manifest(manifest_path)
    splits = read_splits(split_path)
    train_targets = _targets_for_sequences(sequences, splits.get("train", []))
    if train_targets.size == 0:
        msg = "The train split has no TTC targets."
        raise ValueError(msg)

    constants = {
        "mean_train_ttc": float(np.mean(train_targets)),
        "median_train_ttc": float(np.median(train_targets)),
    }
    payload: dict[str, Any] = {
        "baseline": "trivial_constant_ttc",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "train_target_count": int(train_targets.size),
        "predictors": {},
    }

    for predictor_name, constant in constants.items():
        split_metrics: dict[str, Any] = {}
        for split_name, sequence_ids in splits.items():
            targets = _targets_for_sequences(sequences, sequence_ids)
            if targets.size == 0:
                split_metrics[split_name] = {"count": 0, "metrics": None}
                continue
            predictions = np.full_like(targets, fill_value=constant, dtype=np.float64)
            split_metrics[split_name] = {
                "count": int(targets.size),
                "metrics": regression_metrics(targets, predictions),
            }
        payload["predictors"][predictor_name] = {
            "constant_ttc_s": constant,
            "splits": split_metrics,
        }

    if output_path is not None:
        write_structured(output_path, payload)
    return payload
