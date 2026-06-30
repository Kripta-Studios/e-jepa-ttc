"""Supervised ridge baseline using event count/rate features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.evttc import count_events_window, read_manifest
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.utils.io import read_structured, write_structured


def _load_windows(index_path: str | Path) -> list[dict[str, Any]]:
    data = read_structured(index_path)
    windows = data.get("windows")
    if not isinstance(windows, list):
        msg = f"Index {index_path} does not contain a windows list."
        raise ValueError(msg)
    return [dict(item) for item in windows]


def _fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float = 1e-3) -> np.ndarray:
    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def _predict_ridge(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(features.shape[0]), features])
    return design @ beta


def _standardize_train(
    train_features: np.ndarray,
    all_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_features - mean) / std, (all_features - mean) / std, mean, std


def run_event_rate_baseline(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    index_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train/evaluate a ridge regressor on context event-count features."""

    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    splits = read_splits(split_path)
    split_for_sequence = {
        sequence_id: split_name for split_name, ids in splits.items() for sequence_id in ids
    }
    rows: list[dict[str, Any]] = []
    for window in _load_windows(index_path):
        sequence_id = str(window["sequence_id"])
        sequence = sequences[sequence_id]
        event_hdf5 = sequence.resolve("event_hdf5")
        if event_hdf5 is None:
            continue
        start_us = int(window["context_start_us"])
        end_us = int(window["context_end_us"])
        count = count_events_window(event_hdf5, t_start_us=start_us, t_end_us=end_us)
        duration_s = max((end_us - start_us) / 1_000_000.0, 1e-6)
        rows.append(
            {
                "sequence_id": sequence_id,
                "split": split_for_sequence.get(sequence_id, "unassigned"),
                "ttc_seconds": float(window["ttc_seconds"]),
                "features": [
                    float(np.log1p(count)),
                    float(np.log1p(count / duration_s)),
                ],
            }
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        msg = "No train windows available for event-rate baseline."
        raise ValueError(msg)
    all_features = np.array([row["features"] for row in rows], dtype=np.float64)
    train_features = np.array([row["features"] for row in train_rows], dtype=np.float64)
    train_targets = np.log(np.array([row["ttc_seconds"] for row in train_rows], dtype=np.float64))
    train_scaled, all_scaled, mean, std = _standardize_train(train_features, all_features)
    beta = _fit_ridge(train_scaled, train_targets)
    pred_log = _predict_ridge(all_scaled, beta)
    pred_ttc = np.exp(pred_log)

    payload: dict[str, Any] = {
        "baseline": "event_rate_ridge",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "index": Path(index_path).as_posix(),
        "feature_names": ["log_context_event_count", "log_context_event_rate_hz"],
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "beta": beta.tolist(),
        "splits": {},
    }
    for split_name in splits:
        indices = [idx for idx, row in enumerate(rows) if row["split"] == split_name]
        y_true = np.array([rows[idx]["ttc_seconds"] for idx in indices], dtype=np.float64)
        y_pred = pred_ttc[indices]
        payload["splits"][split_name] = {
            "window_count": int(len(indices)),
            "metrics": regression_metrics(y_true, y_pred) if len(indices) else None,
        }
    if output_path is not None:
        write_structured(output_path, payload)
    return payload
