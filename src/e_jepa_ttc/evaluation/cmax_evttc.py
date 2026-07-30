"""Causal raw-event CMax evaluation on labelled EvTTC development splits."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.evttc_object_cache import _states
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.evaluation.strttc_evttc import (
    _causal_roi,
    _event_array_in_roi,
    _sample_indices,
    _sequence_intrinsics,
)
from e_jepa_ttc.geometry.cmax import maximize_radial_event_contrast
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class EvTTCCMaxConfig:
    """Resource-bounded raw-event CMax settings."""

    lookback_s: float = 0.2
    roi_margin_fraction: float = 0.1
    object_event_margin_fraction: float = 0.1
    maximum_samples_per_sequence: int = 8
    maximum_events_per_sample: int = 50_000
    minimum_roi_events: int = 1_000
    minimum_ttc_s: float = 0.25
    maximum_ttc_s: float = 12.0
    coarse_steps: int = 33
    minimum_relative_contrast_gain: float = 0.01


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_raw_events(
    raw: np.ndarray,
    centers_xy: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw.shape[0] <= maximum:
        return raw, centers_xy
    indices = np.linspace(0, raw.shape[0] - 1, maximum, dtype=np.int64)
    return raw[indices], centers_xy[indices]


def _causal_object_events(
    raw: np.ndarray,
    states: list[Any],
    *,
    current_index: int,
    start_us: int,
    roi: tuple[int, int, int, int],
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter events with interpolated past/current boxes and return moving centers."""

    first = next(
        (
            index
            for index, state in enumerate(states[: current_index + 1])
            if state.timestamp_us >= start_us
        ),
        current_index,
    )
    first = max(0, first - 1)
    history = states[first : current_index + 1]
    state_times = np.asarray([state.timestamp_us * 1e-6 for state in history])
    boxes = np.asarray([state.bbox_event_xyxy for state in history], dtype=np.float64)
    event_times = raw[:, 0]
    interpolated = np.column_stack(
        [
            np.interp(event_times, state_times, boxes[:, coordinate])
            for coordinate in range(4)
        ]
    )
    widths = interpolated[:, 2] - interpolated[:, 0]
    heights = interpolated[:, 3] - interpolated[:, 1]
    margin_x = np.maximum(widths * margin_fraction, 1.0)
    margin_y = np.maximum(heights * margin_fraction, 1.0)
    global_x = raw[:, 1] + roi[0]
    global_y = raw[:, 2] + roi[1]
    selected = (
        (global_x >= interpolated[:, 0] - margin_x)
        & (global_x <= interpolated[:, 2] + margin_x)
        & (global_y >= interpolated[:, 1] - margin_y)
        & (global_y <= interpolated[:, 3] + margin_y)
    )
    centers = np.column_stack(
        (
            0.5 * (interpolated[:, 0] + interpolated[:, 2]) - roi[0],
            0.5 * (interpolated[:, 1] + interpolated[:, 3]) - roi[1],
        )
    )
    return raw[selected], centers[selected]


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    result = object_ttc_metrics(truth, prediction)
    result.update(grouped_ttc_selection_components(truth, prediction, groups))
    sequence_mae = [
        float(np.mean(np.abs(truth[groups == group] - prediction[groups == group])))
        for group in np.unique(groups)
    ]
    result["sequence_macro_mae_s"] = float(np.mean(sequence_mae))
    result["worst_sequence_mae_s"] = float(np.max(sequence_mae))
    return result


def evaluate_evttc_cmax(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    split_name: str = "validation",
    config: EvTTCCMaxConfig | None = None,
) -> dict[str, Any]:
    """Evaluate continuous-time radial CMax without using test or future events."""

    resolved = config or EvTTCCMaxConfig()
    assert_no_sealed_benchmark_paths((manifest_path, split_path, output_dir))
    if (
        resolved.lookback_s <= 0.0
        or resolved.maximum_samples_per_sequence <= 0
        or resolved.maximum_events_per_sample <= 0
    ):
        raise ValueError("CMax evaluation budgets must be positive.")
    sequences = {item.sequence_id: item for item in read_manifest(manifest_path)}
    split_ids = read_splits(split_path).get(split_name)
    if not split_ids:
        raise ValueError(f"Split {split_name!r} is absent or empty.")
    lookback_us = int(round(resolved.lookback_s * 1e6))
    truth: list[float] = []
    prediction: list[float] = []
    groups: list[str] = []
    timestamp_rows: list[int] = []
    runtime_rows: list[float] = []
    confidence_rows: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for sequence_id in split_ids:
        sequence = sequences[sequence_id]
        event_path = sequence.resolve("event_hdf5")
        if event_path is None:
            raise FileNotFoundError(f"Missing event HDF5 for {sequence_id}.")
        states = _states(sequence)
        sensor_width, sensor_height, _ = _sequence_intrinsics(event_path)
        indices = _sample_indices(
            states,
            lookback_us=lookback_us,
            maximum=resolved.maximum_samples_per_sequence,
        )
        for sample_index in indices:
            state = states[sample_index]
            start_us = state.timestamp_us - lookback_us
            roi = _causal_roi(
                states,
                current_index=sample_index,
                start_us=start_us,
                width=sensor_width,
                height=sensor_height,
                margin_fraction=resolved.roi_margin_fraction,
            )
            started = time.perf_counter()
            raw = _event_array_in_roi(
                event_path,
                sequence_id=sequence_id,
                start_us=start_us,
                end_us=state.timestamp_us,
                roi=roi,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
            )
            raw_roi_count = int(raw.shape[0])
            raw, event_centers = _causal_object_events(
                raw,
                states,
                current_index=sample_index,
                start_us=start_us,
                roi=roi,
                margin_fraction=resolved.object_event_margin_fraction,
            )
            raw_object_count = int(raw.shape[0])
            if raw_object_count < resolved.minimum_roi_events:
                failure_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "timestamp_us": state.timestamp_us,
                        "reason": f"insufficient_object_events:{raw_object_count}",
                    }
                )
                continue
            raw, event_centers = _bounded_raw_events(
                raw,
                event_centers,
                resolved.maximum_events_per_sample,
            )
            x0, y0, x1, y1 = roi
            current_box = state.bbox_event_xyxy
            center = (
                0.5 * (current_box[0] + current_box[2]) - x0,
                0.5 * (current_box[1] + current_box[3]) - y0,
            )
            result = maximize_radial_event_contrast(
                raw[:, 1:3],
                raw[:, 0],
                raw[:, 3],
                image_shape=(y1 - y0, x1 - x0),
                center_xy=center,
                event_centers_xy=event_centers,
                minimum_ttc_s=resolved.minimum_ttc_s,
                maximum_ttc_s=resolved.maximum_ttc_s,
                coarse_steps=resolved.coarse_steps,
                minimum_events=resolved.minimum_roi_events,
                minimum_relative_contrast_gain=resolved.minimum_relative_contrast_gain,
            )
            elapsed = time.perf_counter() - started
            diagnostics = {
                "sequence_id": sequence_id,
                "timestamp_us": state.timestamp_us,
                "roi": list(roi),
                "raw_roi_events": raw_roi_count,
                "raw_object_events": raw_object_count,
                "optimized_events": int(raw.shape[0]),
                "reason": result.reason,
                "inverse_ttc_per_s": result.inverse_ttc_per_s,
                "contrast": result.contrast,
                "null_contrast": result.null_contrast,
                "relative_contrast_gain": result.relative_contrast_gain,
                "survival_fraction": result.survival_fraction,
                "confidence": result.confidence,
                "evaluations": result.evaluations,
                "runtime_s": elapsed,
            }
            diagnostic_rows.append(diagnostics)
            if not result.valid:
                failure_rows.append(diagnostics)
                continue
            truth.append(float(state.measurement.ttc_seconds))
            prediction.append(float(result.ttc_seconds))
            groups.append(sequence_id)
            timestamp_rows.append(state.timestamp_us)
            runtime_rows.append(elapsed)
            confidence_rows.append(result.confidence)
    y_true = np.asarray(truth, dtype=np.float64)
    y_pred = np.asarray(prediction, dtype=np.float64)
    sequence_rows = np.asarray(groups, dtype=np.str_)
    if y_true.size == 0:
        raise RuntimeError(f"Every CMax sample failed: {failure_rows[:3]}")
    requested = y_true.size + len(failure_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        ttc_true=y_true,
        ttc_pred=y_pred,
        sequence_id=sequence_rows,
        timestamp_us=np.asarray(timestamp_rows, dtype=np.int64),
        runtime_s=np.asarray(runtime_rows, dtype=np.float64),
        confidence=np.asarray(confidence_rows, dtype=np.float64),
    )
    summary: dict[str, Any] = {
        "protocol": "evttc_cmax_radial_source_port_v1_causal",
        "scientific_scope": (
            "Raw-event continuous-time radial CMax with causal GT-box ROI. This is a "
            "declared local adaptation, not an exact reproduction of leaderboard CMax."
        ),
        "source_reference": "Gallego et al., CVPR 2018",
        "config": asdict(resolved),
        "manifest": Path(manifest_path).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "split": Path(split_path).as_posix(),
        "split_sha256": _sha256(split_path),
        "split_name": split_name,
        "git_commit": _git_commit(),
        "successful_sample_metrics": _metrics(y_true, y_pred, sequence_rows),
        "coverage": {
            "requested_samples": int(requested),
            "successful_samples": int(y_true.size),
            "failed_samples": len(failure_rows),
            "success_fraction": float(y_true.size / requested),
            "complete_coverage": not failure_rows,
        },
        "promotion_gate": {
            "eligible": not failure_rows,
            "reason": (
                "Complete coverage; compare numerical gate."
                if not failure_rows
                else "Incomplete coverage; success-only metrics are not promotable."
            ),
        },
        "runtime": {
            "median_s": float(np.median(runtime_rows)),
            "p95_s": float(np.quantile(runtime_rows, 0.95)),
            "mean_s": float(np.mean(runtime_rows)),
        },
        "failure_rows": failure_rows,
        "diagnostics": diagnostic_rows,
        "benchmark10_opened": False,
    }
    write_structured(output / "summary.json", summary)
    return summary


__all__ = ["EvTTCCMaxConfig", "evaluate_evttc_cmax"]
