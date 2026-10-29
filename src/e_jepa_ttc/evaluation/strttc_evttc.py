"""Causal raw-event STRTTC evaluation on labelled EvTTC development splits."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.evttc import read_events_window, read_manifest
from e_jepa_ttc.data.evttc_object_cache import (
    _EvTTCState,
    _normalized_event_intrinsics,
    _states,
)
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.geometry.strttc import (
    inverse_ttc_at_endpoint,
    refine_strttc_on_time_surface,
)
from e_jepa_ttc.geometry.strttc_frontend import (
    STRTTCFrontendConfig,
    run_strttc_linear_frontend,
)
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class EvTTCSTRTTCConfig:
    """Resource-bounded causal adaptation of the official STRTTC source."""

    lookback_s: float = 0.2
    roi_margin_fraction: float = 0.1
    maximum_ttc_s: float = 12.0
    maximum_samples_per_sequence: int = 8
    nonlinear_refinement: bool = False
    nonlinear_maximum_function_evaluations: int = 20
    minimum_roi_events: int = 2_000
    frontend: STRTTCFrontendConfig = STRTTCFrontendConfig()


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


def _sample_indices(
    states: list[_EvTTCState],
    *,
    lookback_us: int,
    maximum: int,
) -> list[int]:
    eligible = [
        index
        for index, state in enumerate(states)
        if state.timestamp_us - states[0].timestamp_us >= lookback_us
    ]
    if len(eligible) <= maximum:
        return eligible
    positions = ((np.arange(maximum, dtype=np.float64) + 0.5) * len(eligible) / maximum).astype(
        np.int64
    )
    return [eligible[int(position)] for position in positions]


def _causal_roi(
    states: list[_EvTTCState],
    *,
    current_index: int,
    start_us: int,
    width: int,
    height: int,
    margin_fraction: float,
) -> tuple[int, int, int, int]:
    history = [
        state.bbox_event_xyxy
        for state in states[: current_index + 1]
        if state.timestamp_us >= start_us
    ]
    boxes = np.asarray(history, dtype=np.float64)
    x0 = float(boxes[:, 0].min())
    y0 = float(boxes[:, 1].min())
    x1 = float(boxes[:, 2].max())
    y1 = float(boxes[:, 3].max())
    margin_x = max((x1 - x0) * margin_fraction, 2.0)
    margin_y = max((y1 - y0) * margin_fraction, 2.0)
    return (
        max(0, int(np.floor(x0 - margin_x))),
        max(0, int(np.floor(y0 - margin_y))),
        min(width, int(np.ceil(x1 + margin_x))),
        min(height, int(np.ceil(y1 + margin_y))),
    )


def _sequence_intrinsics(event_path: Path) -> tuple[int, int, tuple[float, ...]]:
    with h5py.File(event_path, "r") as handle:
        normalized = _normalized_event_intrinsics(handle).astype(np.float64)
    width = 1280
    height = 720
    return (
        width,
        height,
        (
            float(normalized[0] * width),
            float(normalized[1] * height),
            float(normalized[2] * width),
            float(normalized[3] * height),
        ),
    )


def _event_array_in_roi(
    event_path: Path,
    *,
    sequence_id: str,
    start_us: int,
    end_us: int,
    roi: tuple[int, int, int, int],
    sensor_width: int,
    sensor_height: int,
) -> np.ndarray:
    events = read_events_window(
        event_path,
        t_start_us=start_us,
        t_end_us=end_us,
        sequence_id=sequence_id,
        width=sensor_width,
        height=sensor_height,
    )
    x0, y0, x1, y1 = roi
    selected = (events.x >= x0) & (events.x < x1) & (events.y >= y0) & (events.y < y1)
    return np.column_stack(
        (
            events.t_us[selected].astype(np.float64) * 1e-6,
            events.x[selected].astype(np.float64) - x0,
            events.y[selected].astype(np.float64) - y0,
            events.polarity[selected].astype(np.float64),
        )
    )


def evaluate_evttc_strttc(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    split_name: str = "validation",
    config: EvTTCSTRTTCConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a source-traceable raw-event STRTTC port without test access."""

    resolved = config or EvTTCSTRTTCConfig()
    assert_no_sealed_benchmark_paths((manifest_path, split_path, output_dir))
    if (
        resolved.lookback_s <= 0.0
        or resolved.maximum_samples_per_sequence <= 0
        or resolved.minimum_roi_events <= 0
    ):
        raise ValueError("STRTTC evaluation limits must be positive.")
    sequences = {item.sequence_id: item for item in read_manifest(manifest_path)}
    split_ids = read_splits(split_path).get(split_name)
    if not split_ids:
        raise ValueError(f"Split {split_name!r} is absent or empty.")

    truth: list[float] = []
    prediction: list[float] = []
    sequence_rows: list[str] = []
    timestamp_rows: list[int] = []
    runtime_rows: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lookback_us = int(round(resolved.lookback_s * 1e6))
    for sequence_id in split_ids:
        sequence = sequences[sequence_id]
        event_path = sequence.resolve("event_hdf5")
        if event_path is None:
            raise FileNotFoundError(f"Missing event HDF5 for {sequence_id}.")
        states = _states(sequence)
        sensor_width, sensor_height, intrinsics = _sequence_intrinsics(event_path)
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
            try:
                raw = _event_array_in_roi(
                    event_path,
                    sequence_id=sequence_id,
                    start_us=start_us,
                    end_us=state.timestamp_us,
                    roi=roi,
                    sensor_width=sensor_width,
                    sensor_height=sensor_height,
                )
                if raw.shape[0] < resolved.minimum_roi_events:
                    raise RuntimeError(f"Only {raw.shape[0]} ROI events.")
                x0, y0, x1, y1 = roi
                roi_intrinsics = (
                    intrinsics[0],
                    intrinsics[1],
                    intrinsics[2] - x0,
                    intrinsics[3] - y0,
                )
                frontend = replace(
                    resolved.frontend,
                    seed=resolved.frontend.seed + sample_index,
                )
                result = run_strttc_linear_frontend(
                    raw,
                    width=x1 - x0,
                    height=y1 - y0,
                    intrinsics=roi_intrinsics,
                    config=frontend,
                )
                parameters = result.linear.parameters
                if resolved.nonlinear_refinement:
                    parameters = refine_strttc_on_time_surface(
                        parameters,
                        result.contour_txy,
                        result.reference_time_s,
                        result.nearest_linear_time_surface,
                        roi_intrinsics,
                        maximum_function_evaluations=(
                            resolved.nonlinear_maximum_function_evaluations
                        ),
                    )
                q_current = inverse_ttc_at_endpoint(
                    float(parameters[0]),
                    state.timestamp_us * 1e-6 - result.absolute_reference_time_s,
                )
                if q_current <= 0.0 or not np.isfinite(q_current):
                    raise RuntimeError("Non-positive or non-finite inverse TTC.")
                ttc = float(np.clip(1.0 / q_current, 1e-4, resolved.maximum_ttc_s))
                elapsed = time.perf_counter() - started
                truth.append(float(state.measurement.ttc_seconds))
                prediction.append(ttc)
                sequence_rows.append(sequence_id)
                timestamp_rows.append(state.timestamp_us)
                runtime_rows.append(elapsed)
                diagnostics.append(
                    {
                        "sequence_id": sequence_id,
                        "timestamp_us": state.timestamp_us,
                        "roi": list(roi),
                        "roi_events": int(raw.shape[0]),
                        "normal_flows": int(result.normal_flow_xy.shape[0]),
                        "inlier_ratio": result.linear.inlier_ratio,
                        "linear_inverse_ttc_at_reference": result.linear.inverse_ttc,
                        "absolute_reference_time_s": result.absolute_reference_time_s,
                        "inverse_ttc_at_endpoint": q_current,
                    }
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failures.append(
                    {
                        "sequence_id": sequence_id,
                        "timestamp_us": state.timestamp_us,
                        "reason": str(error),
                    }
                )

    y_true = np.asarray(truth, dtype=np.float64)
    y_pred = np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(sequence_rows)
    if y_true.size == 0:
        raise RuntimeError(f"Every STRTTC sample failed: {failures[:3]}")
    successful_sample_metrics = object_ttc_metrics(y_true, y_pred)
    successful_sample_metrics.update(grouped_ttc_selection_components(y_true, y_pred, groups))
    per_sequence_mae = [
        float(np.mean(np.abs(y_true[groups == group] - y_pred[groups == group])))
        for group in np.unique(groups)
    ]
    successful_sample_metrics["sequence_macro_mae_s"] = float(np.mean(per_sequence_mae))
    successful_sample_metrics["worst_sequence_mae_s"] = float(np.max(per_sequence_mae))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        ttc_true=y_true,
        ttc_pred=y_pred,
        sequence_id=groups,
        timestamp_us=np.asarray(timestamp_rows, dtype=np.int64),
        runtime_s=np.asarray(runtime_rows, dtype=np.float64),
    )
    summary = {
        "protocol": "evttc_strttc_source_port_v1_causal",
        "config": asdict(resolved),
        "manifest": Path(manifest_path).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "split": Path(split_path).as_posix(),
        "split_sha256": _sha256(split_path),
        "split_name": split_name,
        "git_commit": _git_commit(),
        "official_source": {
            "repository": "https://github.com/NAIL-HNU/event_aided_ttc",
            "inspected_commit": "79ff0842955304ec4f6164ec09baddc71386d225",
            "ported_stages": [
                "NearestLinearTimeSurfacePositiveAndNegative",
                "GetValidPointonNLTS",
                "PlaneFittingNormalFlowByEventValied",
                "constructA_b",
                "strttc_minimal",
                "str_warping",
                "strttc_optimize",
            ],
            "declared_adaptations": [
                "causal past-only window instead of symmetric offline epoch",
                "native event ROI union from causal projected EvTTC boxes",
                "deterministic bounded RANSAC budgets",
                "median plus Gaussian filter; MATLAB bilateral filter omitted",
            ],
        },
        "successful_sample_metrics": successful_sample_metrics,
        "coverage": {
            "requested_samples": int(y_true.size + len(failures)),
            "successful_samples": int(y_true.size),
            "failed_samples": len(failures),
            "success_fraction": float(y_true.size / (y_true.size + len(failures))),
            "complete_coverage": not failures,
        },
        "promotion_gate": {
            "eligible": False,
            "reason": (
                "The source port has incomplete coverage; success-only metrics cannot "
                "be compared fairly with complete-coverage neural candidates."
                if failures
                else "No promotion threshold was supplied to this diagnostic runner."
            ),
        },
        "successful_samples": int(y_true.size),
        "failed_samples": len(failures),
        "failure_rows": failures,
        "failure_policy": (
            "Failures are neither dropped from a claimed full-set score nor replaced "
            "with ground-truth-aware values. Metrics are explicitly success-only."
        ),
        "diagnostics": diagnostics,
        "runtime": {
            "median_s": float(np.median(runtime_rows)),
            "p95_s": float(np.quantile(runtime_rows, 0.95)),
            "mean_s": float(np.mean(runtime_rows)),
        },
        "benchmark10_opened": False,
    }
    write_structured(output / "summary.json", summary)
    return summary


__all__ = ["EvTTCSTRTTCConfig", "evaluate_evttc_strttc"]
