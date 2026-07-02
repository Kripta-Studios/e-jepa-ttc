"""Detection-assisted TTC baseline using causal events inside bbox/ROI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.annotations import LabelMeasurement, load_measurements_from_manifest
from e_jepa_ttc.data.evttc import read_events_window, read_manifest
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.data.types import DatasetSequence, EventBatch
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.utils.io import write_structured

ROI_EVENT_FEATURE_NAMES: tuple[str, ...] = (
    "log_total_event_count",
    "log_total_event_rate_hz",
    "log_roi_event_count",
    "log_roi_event_rate_hz",
    "roi_event_fraction",
    "roi_positive_fraction",
    "roi_polarity_balance",
    "bbox_area_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "log_bbox_area_pixels",
    "log_roi_event_density",
    "early_late_log_count_delta",
    "roi_time_mean_fraction",
    "roi_time_std_fraction",
    "roi_centroid_x_fraction",
    "roi_centroid_y_fraction",
    "roi_centroid_motion_x_fraction",
    "roi_centroid_motion_y_fraction",
)


@dataclass(frozen=True)
class ROIEventFeatures:
    """Feature vector and audit counters for one bbox/ROI event window."""

    values: tuple[float, ...]
    bbox_xyxy_event: tuple[float, float, float, float]
    roi_event_count: int
    total_event_count: int


@dataclass(frozen=True)
class ROIEventRow:
    """One detection-assisted ROI event TTC training/evaluation row."""

    sequence_id: str
    timestamp_us: int
    ttc_seconds: float
    features: tuple[float, ...]
    bbox_xyxy_event: tuple[float, float, float, float]
    roi_event_count: int
    total_event_count: int
    window_start_us: int
    window_end_us: int


@dataclass(frozen=True)
class ROIRidgeModel:
    """Train-only standardized ridge model for log TTC."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    beta: np.ndarray
    max_ttc_seconds: float


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _log1p_rate(count: int, duration_s: float) -> float:
    return float(np.log1p(count / max(duration_s, 1e-6)))


def _scale_bbox_to_event_plane(
    measurement: LabelMeasurement,
    *,
    event_width: int,
    event_height: int,
) -> tuple[float, float, float, float] | None:
    """Project an RGB label bbox into the event-camera pixel plane."""

    x0, y0, x1, y1 = measurement.bbox_xyxy
    source_width = measurement.image_width or event_width
    source_height = measurement.image_height or event_height
    if source_width <= 0 or source_height <= 0:
        return None

    scale_x = event_width / float(source_width)
    scale_y = event_height / float(source_height)
    scaled = (
        x0 * scale_x,
        y0 * scale_y,
        x1 * scale_x,
        y1 * scale_y,
    )
    clipped = (
        max(0.0, min(float(event_width), scaled[0])),
        max(0.0, min(float(event_height), scaled[1])),
        max(0.0, min(float(event_width), scaled[2])),
        max(0.0, min(float(event_height), scaled[3])),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def extract_roi_event_features(
    events: EventBatch,
    *,
    bbox_xyxy_event: tuple[float, float, float, float],
    reference_time_us: int,
) -> ROIEventFeatures:
    """Extract causal event-count/motion features inside a bbox/ROI."""

    x0, y0, x1, y1 = bbox_xyxy_event
    duration_us = max(reference_time_us - events.t_start_us, 1)
    duration_s = duration_us / 1_000_000.0
    event_area = float(max(events.width * events.height, 1))
    bbox_width = max(x1 - x0, 0.0)
    bbox_height = max(y1 - y0, 0.0)
    bbox_area = max(bbox_width * bbox_height, 1.0)
    bbox_center_x = x0 + bbox_width * 0.5
    bbox_center_y = y0 + bbox_height * 0.5

    causal = events.t_us <= reference_time_us
    total_count = int(np.count_nonzero(causal))
    if total_count:
        x = events.x[causal].astype(np.float64)
        y = events.y[causal].astype(np.float64)
        t = events.t_us[causal].astype(np.int64)
        polarity = events.polarity[causal].astype(np.float64)
    else:
        x = np.empty(0, dtype=np.float64)
        y = np.empty(0, dtype=np.float64)
        t = np.empty(0, dtype=np.int64)
        polarity = np.empty(0, dtype=np.float64)

    roi = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    roi_count = int(np.count_nonzero(roi))
    roi_rate = roi_count / max(duration_s, 1e-6)
    total_rate = total_count / max(duration_s, 1e-6)
    roi_positive_fraction = 0.0
    roi_polarity_balance = 0.0
    early_late_delta = 0.0
    time_mean = 0.0
    time_std = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    centroid_motion_x = 0.0
    centroid_motion_y = 0.0

    if roi_count:
        roi_x = x[roi]
        roi_y = y[roi]
        roi_t = t[roi]
        roi_polarity = polarity[roi]
        roi_positive_fraction = float(np.mean(roi_polarity > 0.0))
        roi_polarity_balance = float(np.mean(roi_polarity))
        elapsed = np.clip((roi_t - events.t_start_us) / max(duration_us, 1), 0.0, 1.0)
        time_mean = float(np.mean(elapsed))
        time_std = float(np.std(elapsed))
        centroid_x = float(np.mean(roi_x) / max(events.width, 1))
        centroid_y = float(np.mean(roi_y) / max(events.height, 1))

        midpoint_us = events.t_start_us + duration_us // 2
        early = roi_t < midpoint_us
        late = ~early
        early_count = int(np.count_nonzero(early))
        late_count = int(np.count_nonzero(late))
        early_late_delta = float(np.log1p(late_count) - np.log1p(early_count))
        if early_count and late_count:
            centroid_motion_x = float((np.mean(roi_x[late]) - np.mean(roi_x[early])) / events.width)
            centroid_motion_y = float(
                (np.mean(roi_y[late]) - np.mean(roi_y[early])) / events.height
            )

    values = (
        float(np.log1p(total_count)),
        float(np.log1p(total_rate)),
        float(np.log1p(roi_count)),
        _log1p_rate(roi_count, duration_s),
        _safe_fraction(roi_count, total_count),
        roi_positive_fraction,
        roi_polarity_balance,
        _safe_fraction(bbox_area, event_area),
        _safe_fraction(bbox_width, events.width),
        _safe_fraction(bbox_height, events.height),
        _safe_fraction(bbox_center_x, events.width),
        _safe_fraction(bbox_center_y, events.height),
        float(np.log1p(bbox_area)),
        float(np.log1p(roi_rate / bbox_area)),
        early_late_delta,
        time_mean,
        time_std,
        centroid_x,
        centroid_y,
        centroid_motion_x,
        centroid_motion_y,
    )
    return ROIEventFeatures(
        values=values,
        bbox_xyxy_event=bbox_xyxy_event,
        roi_event_count=roi_count,
        total_event_count=total_count,
    )


def _fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float) -> np.ndarray:
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


def _feature_matrix(rows: list[ROIEventRow]) -> np.ndarray:
    return np.array([row.features for row in rows], dtype=np.float64)


def _targets(rows: list[ROIEventRow]) -> np.ndarray:
    return np.array([row.ttc_seconds for row in rows], dtype=np.float64)


def _fit_roi_model(
    train_rows: list[ROIEventRow],
    all_rows: list[ROIEventRow],
    *,
    ridge_alpha: float,
    max_ttc_seconds: float,
) -> tuple[ROIRidgeModel, np.ndarray]:
    if not train_rows:
        msg = "Cannot fit ROI event baseline without train rows."
        raise ValueError(msg)
    train_features = _feature_matrix(train_rows)
    all_features = _feature_matrix(all_rows)
    train_scaled, all_scaled, mean, std = _standardize_train(train_features, all_features)
    train_targets = np.log(np.clip(_targets(train_rows), 1e-3, max_ttc_seconds))
    beta = _fit_ridge(train_scaled, train_targets, alpha=ridge_alpha)
    pred_log = _predict_ridge(all_scaled, beta)
    predictions = np.clip(np.exp(pred_log), 1e-3, max_ttc_seconds)
    return (
        ROIRidgeModel(
            feature_mean=mean,
            feature_std=std,
            beta=beta,
            max_ttc_seconds=max_ttc_seconds,
        ),
        predictions,
    )


def _sequence_rows(
    sequence: DatasetSequence,
    measurements: list[LabelMeasurement],
    *,
    context_ms: int,
) -> list[ROIEventRow]:
    event_hdf5 = sequence.resolve("event_hdf5")
    if event_hdf5 is None:
        return []
    context_us = int(context_ms * 1000)
    rows: list[ROIEventRow] = []
    for measurement in sorted(measurements, key=lambda item: item.timestamp_us):
        start_us = int(measurement.timestamp_us - context_us)
        end_us = int(measurement.timestamp_us)
        events = read_events_window(
            event_hdf5,
            t_start_us=start_us,
            t_end_us=end_us,
            sequence_id=sequence.sequence_id,
        )
        bbox = _scale_bbox_to_event_plane(
            measurement,
            event_width=events.width,
            event_height=events.height,
        )
        if bbox is None:
            continue
        extracted = extract_roi_event_features(
            events,
            bbox_xyxy_event=bbox,
            reference_time_us=measurement.timestamp_us,
        )
        rows.append(
            ROIEventRow(
                sequence_id=measurement.sequence_id,
                timestamp_us=measurement.timestamp_us,
                ttc_seconds=measurement.ttc_seconds,
                features=extracted.values,
                bbox_xyxy_event=extracted.bbox_xyxy_event,
                roi_event_count=extracted.roi_event_count,
                total_event_count=extracted.total_event_count,
                window_start_us=start_us,
                window_end_us=end_us,
            )
        )
    return rows


def _count_mean(rows: list[ROIEventRow], field_name: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(getattr(row, field_name)) for row in rows]))


def run_roi_event_baseline(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
    context_ms: int = 100,
    ridge_alpha: float = 1.0,
    max_ttc_seconds: float = 60.0,
    evaluation_splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate a train-only ridge TTC baseline from causal bbox/ROI event features."""

    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    measurements_by_sequence = load_measurements_from_manifest(manifest_path)
    splits = read_splits(split_path)
    if evaluation_splits is None:
        selected_split_names = list(splits)
    else:
        selected_split_names = list(dict.fromkeys(("train", *evaluation_splits)))
        selected_split_names = [name for name in selected_split_names if name in splits]
    rows_by_split: dict[str, list[ROIEventRow]] = {}
    label_count_by_split: dict[str, int] = {}
    all_rows: list[ROIEventRow] = []
    row_splits: list[str] = []

    for split_name in selected_split_names:
        sequence_ids = splits[split_name]
        split_rows: list[ROIEventRow] = []
        total_labels = 0
        for sequence_id in sequence_ids:
            sequence = sequences.get(sequence_id)
            measurements = measurements_by_sequence.get(sequence_id, [])
            total_labels += len(measurements)
            if sequence is None:
                continue
            split_rows.extend(
                _sequence_rows(
                    sequence,
                    measurements,
                    context_ms=context_ms,
                )
            )
        rows_by_split[split_name] = split_rows
        label_count_by_split[split_name] = total_labels
        all_rows.extend(split_rows)
        row_splits.extend([split_name] * len(split_rows))

    train_rows = rows_by_split.get("train", [])
    model, predictions = _fit_roi_model(
        train_rows,
        all_rows,
        ridge_alpha=ridge_alpha,
        max_ttc_seconds=max_ttc_seconds,
    )

    payload: dict[str, Any] = {
        "baseline": "roi_event_ridge",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "context_ms": context_ms,
        "ridge_alpha": ridge_alpha,
        "max_ttc_seconds": max_ttc_seconds,
        "evaluation_splits": selected_split_names,
        "feature_names": list(ROI_EVENT_FEATURE_NAMES),
        "feature_mean": model.feature_mean.tolist(),
        "feature_std": model.feature_std.tolist(),
        "beta": model.beta.tolist(),
        "leakage_audit": {
            "uses_future_events": False,
            "uses_current_bbox": True,
            "uses_future_bboxes": False,
            "uses_validation_or_test_ttc_for_fit": False,
            "event_window": "[timestamp - context_ms, timestamp] only",
            "bbox_protocol": (
                "Detection-assisted bbox/ROI: current object box is used to crop past "
                "events, matching the CMax/STRTTC-style ROI assumption more closely "
                "than full-frame JEPA."
            ),
        },
        "splits": {},
    }
    for split_name, rows in rows_by_split.items():
        indices = [idx for idx, row_split in enumerate(row_splits) if row_split == split_name]
        y_true = _targets(rows)
        y_pred = predictions[indices] if indices else np.empty(0, dtype=np.float64)
        payload["splits"][split_name] = {
            "label_count": label_count_by_split[split_name],
            "prediction_count": int(len(rows)),
            "invalid_roi_count": int(label_count_by_split[split_name] - len(rows)),
            "mean_total_event_count": _count_mean(rows, "total_event_count"),
            "mean_roi_event_count": _count_mean(rows, "roi_event_count"),
            "metrics": regression_metrics(y_true, y_pred) if len(rows) else None,
        }
    if output_path is not None:
        write_structured(output_path, payload)
    return payload
