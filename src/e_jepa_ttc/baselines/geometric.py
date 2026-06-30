"""Geometric TTC baseline from apparent bbox expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.annotations import LabelMeasurement, load_measurements_from_manifest
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.utils.io import write_structured


def _local_scale_derivative(
    timestamps_s: np.ndarray,
    scale: np.ndarray,
    *,
    window: int = 7,
) -> np.ndarray:
    """Estimate local apparent scale derivative by sliding least squares."""

    if window < 3:
        msg = "Derivative window must be at least 3."
        raise ValueError(msg)
    half = window // 2
    derivative = np.full_like(scale, fill_value=np.nan, dtype=np.float64)
    for idx in range(scale.shape[0]):
        start = max(0, idx - half)
        end = min(scale.shape[0], idx + half + 1)
        if end - start < 3:
            continue
        x = timestamps_s[start:end] - timestamps_s[idx]
        y = scale[start:end]
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        derivative[idx] = float(beta[1])
    return derivative


def _predict_sequence(measurements: list[LabelMeasurement]) -> tuple[np.ndarray, np.ndarray]:
    if len(measurements) < 3:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    ordered = sorted(measurements, key=lambda item: item.timestamp_us)
    timestamps_s = np.array([item.timestamp_us / 1_000_000.0 for item in ordered], dtype=np.float64)
    scale = np.array([item.bbox_scale for item in ordered], dtype=np.float64)
    target = np.array([item.ttc_seconds for item in ordered], dtype=np.float64)
    derivative = _local_scale_derivative(timestamps_s, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        prediction = scale / derivative
    valid = np.isfinite(prediction) & (prediction > 0.0) & (prediction < 60.0)
    return target[valid], prediction[valid]


def run_geometric_baseline(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate apparent-expansion TTC baseline on labeled frames."""

    measurements_by_sequence = load_measurements_from_manifest(manifest_path)
    splits = read_splits(split_path)
    payload: dict[str, Any] = {
        "baseline": "geometric_bbox_expansion",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "splits": {},
        "notes": (
            "Diagnostic analytic baseline on frames with ISAT labels only. This implementation "
            "uses a centered derivative window and therefore includes future bboxes; do not use "
            "it for anti-lookahead claims. Use baseline causal-geometry instead."
        ),
    }
    for split_name, sequence_ids in splits.items():
        true_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        total_labels = 0
        for sequence_id in sequence_ids:
            measurements = measurements_by_sequence.get(sequence_id, [])
            total_labels += len(measurements)
            true, pred = _predict_sequence(measurements)
            true_parts.append(true)
            pred_parts.append(pred)
        y_true = np.concatenate(true_parts) if true_parts else np.empty(0, dtype=np.float64)
        y_pred = np.concatenate(pred_parts) if pred_parts else np.empty(0, dtype=np.float64)
        payload["splits"][split_name] = {
            "label_count": total_labels,
            "prediction_count": int(y_true.shape[0]),
            "invalid_prediction_count": int(total_labels - y_true.shape[0]),
            "metrics": regression_metrics(y_true, y_pred) if y_true.size else None,
        }
    if output_path is not None:
        write_structured(output_path, payload)
    return payload
