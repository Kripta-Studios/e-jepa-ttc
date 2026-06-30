"""Causal detection-assisted TTC from apparent bbox expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.annotations import LabelMeasurement, load_measurements_from_manifest
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class CausalGeometryRow:
    """One causal detection-assisted prediction row."""

    sequence_id: str
    timestamp_us: int
    ttc_seconds: float
    raw_ttc_seconds: float
    bbox_scale: float
    scale_derivative: float
    history_count: int


def _causal_scale_derivative(
    timestamps_s: np.ndarray,
    scale: np.ndarray,
    idx: int,
    *,
    window: int,
) -> float | None:
    """Estimate derivative at `idx` using only samples with index <= `idx`."""

    if window < 3:
        msg = "Derivative window must be at least 3."
        raise ValueError(msg)
    start = max(0, idx - window + 1)
    end = idx + 1
    if end - start < 3:
        return None
    x = timestamps_s[start:end] - timestamps_s[idx]
    y = scale[start:end]
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def _sequence_rows(
    measurements: list[LabelMeasurement],
    *,
    window: int,
    max_ttc_seconds: float,
) -> list[CausalGeometryRow]:
    if len(measurements) < 3:
        return []
    ordered = sorted(measurements, key=lambda item: item.timestamp_us)
    timestamps_s = np.array([item.timestamp_us / 1_000_000.0 for item in ordered], dtype=np.float64)
    scale = np.array([item.bbox_scale for item in ordered], dtype=np.float64)
    rows: list[CausalGeometryRow] = []
    for idx, item in enumerate(ordered):
        derivative = _causal_scale_derivative(timestamps_s, scale, idx, window=window)
        if derivative is None or derivative <= 1e-9:
            continue
        prediction = float(scale[idx] / derivative)
        if not np.isfinite(prediction) or not 0.0 < prediction < max_ttc_seconds:
            continue
        rows.append(
            CausalGeometryRow(
                sequence_id=item.sequence_id,
                timestamp_us=item.timestamp_us,
                ttc_seconds=item.ttc_seconds,
                raw_ttc_seconds=prediction,
                bbox_scale=float(scale[idx]),
                scale_derivative=derivative,
                history_count=idx + 1,
            )
        )
    return rows


def _fit_log_affine(rows: list[CausalGeometryRow]) -> list[float]:
    if not rows:
        msg = "Cannot fit causal geometry calibration without train rows."
        raise ValueError(msg)
    x = np.log(np.array([row.raw_ttc_seconds for row in rows], dtype=np.float64))
    y = np.log(np.array([row.ttc_seconds for row in rows], dtype=np.float64))
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return [float(value) for value in beta]


def _predict_calibrated(rows: list[CausalGeometryRow], beta: list[float]) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float64)
    x = np.log(np.array([row.raw_ttc_seconds for row in rows], dtype=np.float64))
    return np.exp(float(beta[0]) + float(beta[1]) * x)


def _targets(rows: list[CausalGeometryRow]) -> np.ndarray:
    return np.array([row.ttc_seconds for row in rows], dtype=np.float64)


def run_causal_geometry_baseline(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
    derivative_window: int = 15,
    max_ttc_seconds: float = 60.0,
) -> dict[str, Any]:
    """Evaluate causal bbox-expansion TTC with train-only log-affine calibration."""

    measurements_by_sequence = load_measurements_from_manifest(manifest_path)
    splits = read_splits(split_path)
    rows_by_split: dict[str, list[CausalGeometryRow]] = {}
    label_count_by_split: dict[str, int] = {}
    for split_name, sequence_ids in splits.items():
        rows: list[CausalGeometryRow] = []
        total_labels = 0
        for sequence_id in sequence_ids:
            measurements = measurements_by_sequence.get(sequence_id, [])
            total_labels += len(measurements)
            rows.extend(
                _sequence_rows(
                    measurements,
                    window=derivative_window,
                    max_ttc_seconds=max_ttc_seconds,
                )
            )
        rows_by_split[split_name] = rows
        label_count_by_split[split_name] = total_labels

    train_rows = rows_by_split.get("train", [])
    calibration_beta = _fit_log_affine(train_rows)
    payload: dict[str, Any] = {
        "baseline": "causal_detection_assisted_geometry",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "derivative_window": derivative_window,
        "max_ttc_seconds": max_ttc_seconds,
        "calibration": {
            "type": "train_only_log_affine",
            "beta": calibration_beta,
            "train_prediction_count": len(train_rows),
        },
        "leakage_audit": {
            "uses_future_events": False,
            "uses_future_bboxes": False,
            "uses_validation_or_test_ttc_for_fit": False,
            "bbox_protocol": (
                "Detection-assisted: requires current/past object boxes at inference; "
                "do not compare as event-only."
            ),
        },
        "splits": {},
    }
    for split_name, rows in rows_by_split.items():
        y_true = _targets(rows)
        y_raw = np.array([row.raw_ttc_seconds for row in rows], dtype=np.float64)
        y_calibrated = _predict_calibrated(rows, calibration_beta)
        payload["splits"][split_name] = {
            "label_count": label_count_by_split[split_name],
            "prediction_count": int(len(rows)),
            "invalid_prediction_count": int(label_count_by_split[split_name] - len(rows)),
            "raw_metrics": regression_metrics(y_true, y_raw) if y_true.size else None,
            "metrics": regression_metrics(y_true, y_calibrated) if y_true.size else None,
        }
    if output_path is not None:
        write_structured(output_path, payload)
    return payload
