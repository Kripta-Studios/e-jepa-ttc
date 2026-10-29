"""Causal detection-assisted TTC from apparent bbox expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.annotations import LabelMeasurement, load_measurements_from_manifest
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
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


def _hybrid_geometry_with_fallback(
    geometry_predictions: np.ndarray,
    neural_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use causal geometry when finite and a neural prediction otherwise."""

    geometry = np.asarray(geometry_predictions, dtype=np.float64).reshape(-1)
    neural = np.asarray(neural_predictions, dtype=np.float64).reshape(-1)
    if geometry.shape != neural.shape:
        raise ValueError("Geometry and neural predictions must be shape matched.")
    valid_geometry = np.isfinite(geometry) & (geometry > 0.0)
    hybrid = np.where(valid_geometry, geometry, neural)
    return hybrid, valid_geometry


def _cache_rows(
    cache_manifest_path: str | Path,
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = EAPObjectCacheDataset(cache_manifest_path, splits=(split,))
    sequences: list[str] = []
    timestamps_us: list[int] = []
    targets: list[float] = []
    sample_tokens: list[str] = []
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            context_end = sample["context_window_end_us"]
            ttc = sample["ttc_s"]
            if isinstance(context_end, str) or isinstance(ttc, str):
                raise TypeError("Cache tensors were unexpectedly serialized as strings.")
            sequences.append(str(sample["sequence_id"]))
            timestamps_us.append(int(context_end.reshape(-1)[-1].item()))
            targets.append(float(ttc.reshape(-1)[0].item()))
            sample_tokens.append(str(sample["sample_token"]))
    finally:
        dataset.close()
    return (
        np.asarray(sequences),
        np.asarray(timestamps_us, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(sample_tokens),
    )


def _grouped_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, object]:
    return {
        **object_ttc_metrics(targets, predictions),
        "grouped_checkpoint_selection": grouped_ttc_selection_components(
            targets,
            predictions,
            sequences,
        ),
    }


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


def run_cache_aligned_causal_geometry_hybrid(
    *,
    manifest_path: str | Path,
    cache_manifest_path: str | Path,
    neural_predictions_path: str | Path,
    output_path: str | Path | None = None,
    derivative_window: int = 21,
    max_ttc_seconds: float = 60.0,
) -> dict[str, Any]:
    """Evaluate calibrated causal bbox geometry with a neural invalid-row fallback.

    Calibration is fit only on cache-aligned train rows. Validation geometry
    uses current and past ground-truth boxes; the neural model is used only
    where the physical estimator cannot produce a finite positive value.
    """

    measurements_by_sequence = load_measurements_from_manifest(manifest_path)
    geometry_by_key: dict[tuple[str, int], CausalGeometryRow] = {}
    for measurements in measurements_by_sequence.values():
        for row in _sequence_rows(
            measurements,
            window=derivative_window,
            max_ttc_seconds=max_ttc_seconds,
        ):
            geometry_by_key[(row.sequence_id, row.timestamp_us)] = row

    train_sequences, train_timestamps, train_targets, _ = _cache_rows(
        cache_manifest_path,
        split="train",
    )
    aligned_train_rows: list[CausalGeometryRow] = []
    for sequence, timestamp, target in zip(
        train_sequences,
        train_timestamps,
        train_targets,
        strict=True,
    ):
        row = geometry_by_key.get((str(sequence), int(timestamp)))
        if row is not None:
            if not np.isclose(row.ttc_seconds, target, rtol=1e-5, atol=1e-5):
                raise ValueError("Train cache TTC does not match the source annotation.")
            aligned_train_rows.append(row)
    calibration_beta = _fit_log_affine(aligned_train_rows)

    val_sequences, val_timestamps, val_targets, val_tokens = _cache_rows(
        cache_manifest_path,
        split="validation",
    )
    geometry_predictions = np.full(val_targets.shape, np.nan, dtype=np.float64)
    for index, (sequence, timestamp, target) in enumerate(
        zip(val_sequences, val_timestamps, val_targets, strict=True)
    ):
        row = geometry_by_key.get((str(sequence), int(timestamp)))
        if row is None:
            continue
        if not np.isclose(row.ttc_seconds, target, rtol=1e-5, atol=1e-5):
            raise ValueError("Validation cache TTC does not match the source annotation.")
        geometry_predictions[index] = _predict_calibrated([row], calibration_beta)[0]

    source = Path(neural_predictions_path)
    with np.load(source, allow_pickle=False) as predictions:
        neural_targets = np.asarray(predictions["ttc_true"], dtype=np.float64)
        neural_predictions = np.asarray(predictions["ttc_pred"], dtype=np.float64)
        neural_sequences = np.asarray(predictions["sequence_id"]).astype(str)
    if not np.allclose(neural_targets, val_targets, rtol=1e-5, atol=1e-5):
        raise ValueError("Neural prediction targets do not match validation cache order.")
    if not np.array_equal(neural_sequences, val_sequences):
        raise ValueError("Neural prediction sequence order does not match validation cache.")

    hybrid_predictions, valid_geometry = _hybrid_geometry_with_fallback(
        geometry_predictions,
        neural_predictions,
    )
    valid_count = int(np.count_nonzero(valid_geometry))
    payload: dict[str, Any] = {
        "experiment": "causal_bbox_geometry_with_neural_invalid_fallback",
        "manifest": Path(manifest_path).as_posix(),
        "cache_manifest": Path(cache_manifest_path).as_posix(),
        "neural_predictions": source.as_posix(),
        "derivative_window": derivative_window,
        "max_ttc_seconds": max_ttc_seconds,
        "calibration": {
            "type": "train_only_log_affine",
            "beta": calibration_beta,
            "cache_train_rows": int(train_targets.size),
            "matched_valid_train_rows": len(aligned_train_rows),
        },
        "coverage": {
            "validation_rows": int(val_targets.size),
            "valid_geometry_rows": valid_count,
            "neural_fallback_rows": int(val_targets.size - valid_count),
            "geometry_fraction": float(valid_count / max(val_targets.size, 1)),
        },
        "comparison": {
            "neural": _grouped_metrics(
                val_targets,
                neural_predictions,
                val_sequences,
            ),
            "hybrid": _grouped_metrics(
                val_targets,
                hybrid_predictions,
                val_sequences,
            ),
            "geometry_valid_rows_only": (
                _grouped_metrics(
                    val_targets[valid_geometry],
                    geometry_predictions[valid_geometry],
                    val_sequences[valid_geometry],
                )
                if valid_count
                else None
            ),
        },
        "leakage_audit": {
            "uses_future_events": False,
            "uses_future_bboxes": False,
            "uses_validation_ttc_for_calibration": False,
            "fallback_rule_tuned_on_validation": False,
            "derivative_window_selected_on_this_development_split": True,
            "benchmark10_opened": False,
        },
        "protocol_disclosure": {
            "track": "bbox_assisted",
            "geometry_input": "current_and_past_ground_truth_bbox_scale",
            "additional_context_vs_neural_model": (
                f"Up to {derivative_window} causal bbox observations versus the "
                "neural cache's three event frames."
            ),
            "fallback": "Use the neural prediction only when geometry is invalid.",
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        write_structured(destination, payload)
        predictions_path = destination.with_name(f"{destination.stem}_predictions.npz")
        np.savez_compressed(
            predictions_path,
            sample_token=val_tokens,
            sequence_id=val_sequences,
            timestamp_us=val_timestamps,
            ttc_true=val_targets.astype(np.float32),
            ttc_neural=neural_predictions.astype(np.float32),
            ttc_geometry=geometry_predictions.astype(np.float32),
            geometry_valid=valid_geometry,
            ttc_hybrid=hybrid_predictions.astype(np.float32),
        )
        payload["predictions_path"] = predictions_path.as_posix()
        write_structured(destination, payload)
    return payload
