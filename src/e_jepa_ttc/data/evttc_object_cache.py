"""Object-centric Event-JEPA cache for local EvTTC sequences and ego motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.annotations import LabelMeasurement, load_label_measurements
from e_jepa_ttc.data.eap_cache import _write_shard
from e_jepa_ttc.data.evttc import (
    HDF5EventLayout,
    _refine_bounds,
    _slice_bounds_from_ms_map,
    discover_event_layout,
    normalize_polarity,
    read_manifest,
)
from e_jepa_ttc.data.targets import TTCTable, load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence, EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class EvTTCObjectCacheConfig:
    """Causal EvTTC cache parameters."""

    history_frames: int = 3
    prediction_horizons_ms: tuple[int, ...] = (100, 250, 500)
    event_window_ms: int = 100
    maximum_target_slop_ms: int = 30
    maximum_history_gap_ms: int = 80
    width: int = 160
    height: int = 90
    event_bins: int = 5
    normalize_events: bool = True
    shard_size: int = 256
    roi_expansion: float = 1.25
    action_dim: int = 8

    def __post_init__(self) -> None:
        if self.history_frames < 2 or self.event_window_ms <= 0:
            msg = "history_frames must be >=2 and event_window_ms positive."
            raise ValueError(msg)
        if any(horizon < self.event_window_ms for horizon in self.prediction_horizons_ms):
            msg = "Future event targets must not overlap the causal context window."
            raise ValueError(msg)
        if tuple(sorted(set(self.prediction_horizons_ms))) != self.prediction_horizons_ms:
            msg = "prediction_horizons_ms must be unique and increasing."
            raise ValueError(msg)
        if min(self.width, self.height, self.event_bins, self.shard_size) <= 0:
            msg = "Spatial, bin and shard dimensions must be positive."
            raise ValueError(msg)
        if self.action_dim != 8:
            msg = "EvTTC egoaction contract contains exactly eight physical features."
            raise ValueError(msg)


@dataclass(frozen=True)
class _EvTTCState:
    measurement: LabelMeasurement
    bbox_event_xyxy: tuple[float, float, float, float]
    depth_m: float

    @property
    def timestamp_us(self) -> int:
        return self.measurement.timestamp_us


@dataclass(frozen=True)
class _CrossCameraCalibration:
    blackfly_intrinsics: np.ndarray
    blackfly_distortion: np.ndarray
    event_intrinsics: np.ndarray
    event_distortion: np.ndarray
    event_from_blackfly: np.ndarray


def _depth_at(table: TTCTable, timestamp_us: int) -> float:
    return float(np.interp(timestamp_us * 1e-6, table["timestamp_s"], table["distance"]))


def _load_cross_camera_calibration(event_path: Path) -> _CrossCameraCalibration | None:
    names = {
        "blackfly_intrinsics": "blackflys/left/calib/intrinsics",
        "blackfly_distortion": "blackflys/left/calib/distortion_coeffs",
        "event_intrinsics": "prophesee/event_cam_left/calib/intrinsics",
        "event_distortion": "prophesee/event_cam_left/calib/distortion_coeffs",
        "event_from_blackfly": "prophesee/event_cam_left/calib/T_bfs_to_prophesee",
    }
    with h5py.File(event_path, "r") as handle:
        if any(path not in handle for path in names.values()):
            return None
        arrays = {key: np.asarray(handle[path], dtype=np.float64) for key, path in names.items()}
    return _CrossCameraCalibration(**arrays)


def _undistort_radtan(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    distorted = np.asarray(points, dtype=np.float64)
    estimate = distorted.copy()
    k1, k2, p1, p2 = np.asarray(coefficients, dtype=np.float64)[:4]
    for _ in range(8):
        x, y = estimate[:, 0], estimate[:, 1]
        radius_squared = x * x + y * y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
        tangent_x = 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x**2)
        tangent_y = p1 * (radius_squared + 2.0 * y**2) + 2.0 * p2 * x * y
        estimate[:, 0] = (distorted[:, 0] - tangent_x) / radial
        estimate[:, 1] = (distorted[:, 1] - tangent_y) / radial
    return estimate


def _distort_radtan(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    x, y = values[:, 0], values[:, 1]
    k1, k2, p1, p2 = np.asarray(coefficients, dtype=np.float64)[:4]
    radius_squared = x * x + y * y
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    return np.column_stack(
        (
            x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x**2),
            y * radial + p1 * (radius_squared + 2.0 * y**2) + 2.0 * p2 * x * y,
        )
    )


def _event_box(
    measurement: LabelMeasurement,
    *,
    depth_m: float | None = None,
    calibration: _CrossCameraCalibration | None = None,
) -> tuple[float, float, float, float] | None:
    """Project a Blackfly bbox to the event camera, with a scaling fallback."""

    source_width = measurement.image_width or 1280
    source_height = measurement.image_height or 720
    if source_width <= 0 or source_height <= 0:
        return None
    x0, y0, x1, y1 = measurement.bbox_xyxy
    if calibration is not None and depth_m is not None and depth_m > 0:
        blackfly = calibration.blackfly_intrinsics
        event = calibration.event_intrinsics
        pixels = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)
        normalized_distorted = np.column_stack(
            ((pixels[:, 0] - blackfly[2]) / blackfly[0], (pixels[:, 1] - blackfly[3]) / blackfly[1])
        )
        normalized = _undistort_radtan(normalized_distorted, calibration.blackfly_distortion)
        points_blackfly = np.column_stack(
            (
                normalized[:, 0] * depth_m,
                normalized[:, 1] * depth_m,
                np.full(4, depth_m),
                np.ones(4),
            )
        )
        points_event = (calibration.event_from_blackfly @ points_blackfly.T).T[:, :3]
        if np.all(points_event[:, 2] > 0):
            event_normalized = points_event[:, :2] / points_event[:, 2:3]
            event_distorted = _distort_radtan(
                event_normalized,
                calibration.event_distortion,
            )
            event_pixels = np.column_stack(
                (
                    event_distorted[:, 0] * event[0] + event[2],
                    event_distorted[:, 1] * event[1] + event[3],
                )
            )
            scaled = (
                float(event_pixels[:, 0].min()),
                float(event_pixels[:, 1].min()),
                float(event_pixels[:, 0].max()),
                float(event_pixels[:, 1].max()),
            )
        else:
            scaled = (0.0, 0.0, 0.0, 0.0)
    else:
        scaled = (
            x0 * 1280.0 / source_width,
            y0 * 720.0 / source_height,
            x1 * 1280.0 / source_width,
            y1 * 720.0 / source_height,
        )
    clipped = (
        float(np.clip(scaled[0], 0.0, 1280.0)),
        float(np.clip(scaled[1], 0.0, 720.0)),
        float(np.clip(scaled[2], 0.0, 1280.0)),
        float(np.clip(scaled[3], 0.0, 720.0)),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _normalized_box(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(box, dtype=np.float32) / np.asarray(
        [1280.0, 720.0, 1280.0, 720.0],
        dtype=np.float32,
    )


def _downsample(events: EventBatch, *, width: int, height: int) -> EventBatch:
    if events.num_events == 0:
        return EventBatch.empty(
            width=width,
            height=height,
            sequence_id=events.sequence_id,
            t_start_us=events.t_start_us,
            t_end_us=events.t_end_us,
        )
    x = np.clip(
        np.floor(events.x.astype(np.float64) * width / events.width).astype(np.int32),
        0,
        width - 1,
    )
    y = np.clip(
        np.floor(events.y.astype(np.float64) * height / events.height).astype(np.int32),
        0,
        height - 1,
    )
    return EventBatch(
        x=x,
        y=y,
        t_us=events.t_us,
        polarity=events.polarity,
        width=width,
        height=height,
        sequence_id=events.sequence_id,
        t_start_us=events.t_start_us,
        t_end_us=events.t_end_us,
    )


def _read_events(
    handle: h5py.File,
    layout: HDF5EventLayout,
    sequence_id: str,
    *,
    start_us: int,
    end_us: int,
) -> EventBatch:
    if layout.kind != "separate" or not all((layout.x, layout.y, layout.t, layout.p)):
        msg = "EvTTC object cache currently requires separate x/y/t/p event arrays."
        raise ValueError(msg)
    assert layout.x and layout.y and layout.t and layout.p
    event_count = int(handle[layout.t].shape[0])
    if layout.ms_map_idx and layout.ms_map_idx in handle:
        rough_start, rough_end = _slice_bounds_from_ms_map(
            handle[layout.ms_map_idx],
            event_count=event_count,
            t_start_us=start_us,
            t_end_us=end_us,
        )
    else:
        rough_start, rough_end = 0, event_count
    timestamps = handle[layout.t][rough_start:rough_end]
    first, last = _refine_bounds(timestamps, rough_start, start_us, end_us)
    return EventBatch(
        x=handle[layout.x][first:last].astype(np.int32),
        y=handle[layout.y][first:last].astype(np.int32),
        t_us=handle[layout.t][first:last].astype(np.int64),
        polarity=normalize_polarity(handle[layout.p][first:last]),
        width=int(layout.width or 1280),
        height=int(layout.height or 720),
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )


def _voxel(
    handle: h5py.File,
    layout: HDF5EventLayout,
    sequence_id: str,
    *,
    start_us: int,
    end_us: int,
    config: EvTTCObjectCacheConfig,
) -> np.ndarray:
    events = _read_events(
        handle,
        layout,
        sequence_id,
        start_us=start_us,
        end_us=end_us,
    )
    return encode_voxel_grid(
        _downsample(events, width=config.width, height=config.height),
        bins=config.event_bins,
        normalize=config.normalize_events,
    ).astype(np.float16)


def _actions(
    handle: h5py.File,
    *,
    start_us: int,
    end_us: int,
) -> tuple[np.ndarray, bool]:
    features = np.zeros(8, dtype=np.float32)
    base = "integratedNavigation/data"
    required = [f"{base}/ts", f"{base}/velocity", f"{base}/attitude"]
    if any(name not in handle for name in required):
        return features, False
    timestamps = handle[f"{base}/ts"]
    start = int(np.searchsorted(timestamps, start_us, side="left"))
    end = int(np.searchsorted(timestamps, end_us, side="right"))
    if end <= start:
        causal_end = int(np.searchsorted(timestamps, end_us, side="right"))
        if causal_end <= 0:
            return features, False
        start, end = max(0, causal_end - 1), causal_end
    times = timestamps[start:end].astype(np.int64)
    velocity = handle[f"{base}/velocity"][start:end].astype(np.float32)
    attitude = handle[f"{base}/attitude"][start:end].astype(np.float32)
    if velocity.size == 0:
        return features, False
    features[0] = np.float32(np.linalg.norm(velocity[-1]))
    features[1:4] = velocity[-1]
    if velocity.shape[0] >= 2:
        duration_s = max(float(times[-1] - times[0]) * 1e-6, 1e-6)
        features[4:7] = (velocity[-1] - velocity[0]) / duration_s
        features[7] = (attitude[-1, 2] - attitude[0, 2]) / duration_s
    return features, True


def _states(sequence: DatasetSequence) -> list[_EvTTCState]:
    ttc_path = sequence.resolve("ttc_csv")
    if ttc_path is None:
        return []
    table = load_ttc_csv(ttc_path)
    event_path = sequence.resolve("event_hdf5")
    calibration = _load_cross_camera_calibration(event_path) if event_path is not None else None
    states: list[_EvTTCState] = []
    for measurement in load_label_measurements(sequence):
        depth = _depth_at(table, measurement.timestamp_us)
        box = _event_box(measurement, depth_m=depth, calibration=calibration)
        if box is not None and np.isfinite(measurement.ttc_seconds):
            states.append(
                _EvTTCState(
                    measurement=measurement,
                    bbox_event_xyxy=box,
                    depth_m=depth,
                )
            )
    return sorted(states, key=lambda state: state.timestamp_us)


def _windows(
    states: list[_EvTTCState],
    config: EvTTCObjectCacheConfig,
) -> list[tuple[tuple[_EvTTCState, ...], dict[int, _EvTTCState]]]:
    timestamps = np.asarray([state.timestamp_us for state in states], dtype=np.int64)
    windows: list[tuple[tuple[_EvTTCState, ...], dict[int, _EvTTCState]]] = []
    for index in range(config.history_frames - 1, len(states)):
        history = tuple(states[index - config.history_frames + 1 : index + 1])
        history_gaps = np.diff([state.timestamp_us for state in history])
        if np.any(history_gaps > config.maximum_history_gap_ms * 1000):
            continue
        future: dict[int, _EvTTCState] = {}
        for horizon_ms in config.prediction_horizons_ms:
            desired = history[-1].timestamp_us + horizon_ms * 1000
            insertion = int(np.searchsorted(timestamps, desired))
            candidates = [
                candidate
                for candidate in (insertion - 1, insertion)
                if index < candidate < len(states)
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda candidate: abs(int(timestamps[candidate]) - desired))
            if abs(int(timestamps[best]) - desired) <= config.maximum_target_slop_ms * 1000:
                future[horizon_ms] = states[best]
        if future:
            windows.append((history, future))
    return windows


def _sample(
    sequence: DatasetSequence,
    handle: h5py.File,
    layout: HDF5EventLayout,
    split: str,
    history: tuple[_EvTTCState, ...],
    future: dict[int, _EvTTCState],
    config: EvTTCObjectCacheConfig,
) -> dict[str, np.ndarray | str]:
    event_path = sequence.resolve("event_hdf5")
    if event_path is None:
        raise ValueError(f"Sequence {sequence.sequence_id} has no event HDF5 path.")
    context_events: list[np.ndarray] = []
    context_actions: list[np.ndarray] = []
    context_action_mask: list[bool] = []
    context_start: list[int] = []
    for state in history:
        start_us = state.timestamp_us - config.event_window_ms * 1000
        context_events.append(
            _voxel(
                handle,
                layout,
                sequence.sequence_id,
                start_us=start_us,
                end_us=state.timestamp_us,
                config=config,
            )
        )
        action, valid = _actions(handle, start_us=start_us, end_us=state.timestamp_us)
        context_actions.append(action)
        context_action_mask.append(valid)
        context_start.append(start_us)

    channel_count = config.event_bins * 2
    future_events: list[np.ndarray] = []
    future_boxes: list[np.ndarray] = []
    future_depth: list[float] = []
    future_mask: list[bool] = []
    future_actions: list[np.ndarray] = []
    future_action_mask: list[bool] = []
    future_start: list[int] = []
    future_end: list[int] = []
    target_time = history[-1].timestamp_us
    for horizon_ms in config.prediction_horizons_ms:
        state = future.get(horizon_ms)
        if state is None:
            future_events.append(
                np.zeros((channel_count, config.height, config.width), dtype=np.float16)
            )
            future_boxes.append(np.zeros(4, dtype=np.float32))
            future_depth.append(float("nan"))
            future_mask.append(False)
            future_actions.append(np.zeros(config.action_dim, dtype=np.float32))
            future_action_mask.append(False)
            future_start.append(-1)
            future_end.append(-1)
            continue
        start_us = max(target_time, state.timestamp_us - config.event_window_ms * 1000)
        future_events.append(
            _voxel(
                handle,
                layout,
                sequence.sequence_id,
                start_us=start_us,
                end_us=state.timestamp_us,
                config=config,
            )
        )
        action, valid = _actions(handle, start_us=target_time, end_us=state.timestamp_us)
        future_boxes.append(_normalized_box(state.bbox_event_xyxy))
        future_depth.append(state.depth_m)
        future_mask.append(True)
        future_actions.append(action)
        future_action_mask.append(valid)
        future_start.append(start_us)
        future_end.append(state.timestamp_us)

    boxes = np.stack([_normalized_box(state.bbox_event_xyxy) for state in history])[:, None]
    return {
        "context_events": np.stack(context_events),
        "context_boxes": boxes,
        "context_sampling_boxes": boxes.copy(),
        "context_object_mask": np.ones((config.history_frames, 1), dtype=np.bool_),
        "context_depth_m": np.asarray([history[-1].depth_m], dtype=np.float32),
        "context_depth_history_m": np.asarray(
            [state.depth_m for state in history],
            dtype=np.float32,
        )[:, None],
        "context_ego_actions": np.stack(context_actions),
        "context_ego_action_mask": np.asarray(context_action_mask, dtype=np.bool_),
        "future_events": np.stack(future_events),
        "future_boxes": np.stack(future_boxes)[:, None],
        "future_sampling_boxes": np.stack(future_boxes)[:, None],
        "future_object_mask": np.asarray(future_mask, dtype=np.bool_)[:, None],
        "future_depth_m": np.asarray(future_depth, dtype=np.float32)[:, None],
        "future_ego_actions": np.stack(future_actions),
        "future_ego_action_mask": np.asarray(future_action_mask, dtype=np.bool_),
        "ttc_s": np.asarray([history[-1].measurement.ttc_seconds], dtype=np.float32),
        "context_window_start_us": np.asarray(context_start, dtype=np.int64),
        "context_window_end_us": np.asarray(
            [state.timestamp_us for state in history],
            dtype=np.int64,
        ),
        "future_window_start_us": np.asarray(future_start, dtype=np.int64),
        "future_window_end_us": np.asarray(future_end, dtype=np.int64),
        "sample_token": f"{sequence.sequence_id}:{history[-1].measurement.frame_index}",
        "sequence_id": sequence.sequence_id,
        "track_id": f"{sequence.sequence_id}:largest_labelled_object",
        "category": history[-1].measurement.category,
        "split": split,
        "ttc_source": "official_evttc_interpolated_ttc_table",
    }


def materialize_evttc_object_cache(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    sequence_splits: dict[str, str],
    config: EvTTCObjectCacheConfig | None = None,
    max_windows_per_sequence: int | None = None,
) -> dict[str, Any]:
    """Build full-frame object caches with causal measured ego motion."""

    config = config or EvTTCObjectCacheConfig()
    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    missing = sorted(set(sequence_splits) - set(sequences))
    if missing:
        raise ValueError(f"EvTTC sequence assignments not in manifest: {missing}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for sequence_id, split in sorted(sequence_splits.items()):
        sequence = sequences[sequence_id]
        states = _states(sequence)
        windows = _windows(states, config)
        if max_windows_per_sequence is not None and len(windows) > max_windows_per_sequence:
            indices = np.linspace(0, len(windows) - 1, max_windows_per_sequence, dtype=int)
            windows = [windows[int(index)] for index in indices]
        pending: list[dict[str, np.ndarray | str]] = []
        shard_index = 0
        event_path = sequence.resolve("event_hdf5")
        if event_path is None:
            raise ValueError(f"Sequence {sequence_id} has no event HDF5 path.")
        layout = discover_event_layout(event_path)
        if layout is None:
            raise ValueError(f"Could not discover event layout for {sequence_id}.")
        with h5py.File(event_path, "r") as handle:
            for history, future in windows:
                pending.append(_sample(sequence, handle, layout, split, history, future, config))
                if len(pending) >= config.shard_size:
                    shards.append(
                        _write_shard(
                            output,
                            sequence_id=sequence_id,
                            split_name=split,
                            shard_index=shard_index,
                            samples=pending,
                            config=config,  # type: ignore[arg-type]
                        )
                    )
                    pending = []
                    shard_index += 1
        if pending:
            shards.append(
                _write_shard(
                    output,
                    sequence_id=sequence_id,
                    split_name=split,
                    shard_index=shard_index,
                    samples=pending,
                    config=config,  # type: ignore[arg-type]
                )
            )
        summaries.append(
            {
                "sequence_id": sequence_id,
                "split": split,
                "labelled_states": len(states),
                "windows": len(windows),
            }
        )
    payload: dict[str, Any] = {
        "format": "evttc_object_event_jepa_cache_v1",
        "manifest": Path(manifest_path).as_posix(),
        "pre_cropped_events": False,
        "config": asdict(config),
        "sequence_splits": sequence_splits,
        "ttc_label_status": "official_evttc_ttc_table_interpolated_at_label_timestamp",
        "ego_action_status": "causal_integrated_navigation_eight_features",
        "future_teacher_uses_ego_actions": False,
        "bbox_alignment": "depth_assisted_radtan_blackfly_to_prophesee_calibrated_projection",
        "normalization": "occupied_voxel_noncentred_q95_magnitude",
        "sequences": summaries,
        "shards": shards,
        "total_samples": sum(int(shard["samples"]) for shard in shards),
        "total_size_bytes": sum(int(shard["size_bytes"]) for shard in shards),
    }
    write_structured(output / "manifest.json", payload)
    return payload


__all__ = ["EvTTCObjectCacheConfig", "materialize_evttc_object_cache"]
