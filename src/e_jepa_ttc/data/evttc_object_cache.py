"""Object-centric Event-JEPA cache for local EvTTC sequences and ego motion."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

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
from e_jepa_ttc.data.storage_guard import (
    StorageBudget,
    assert_bounded_cache_request,
    assert_storage_budget,
    directory_size_bytes,
    estimate_dense_voxel_cache_bytes,
)
from e_jepa_ttc.data.targets import TTCTable, load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence, EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class EvTTCObjectCacheConfig:
    """Causal EvTTC cache parameters."""

    history_frames: int = 3
    history_stride_frames: int = 1
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
    include_rgb: bool = True
    rgb_width: int = 128
    rgb_height: int = 128
    include_segmentation_masks: bool = True
    include_context_events: bool = True
    include_future_events: bool = True
    include_garl_pair: bool = True
    garl_time_surface_planes: int = 20

    def __post_init__(self) -> None:
        if (
            self.history_frames < 2
            or self.history_stride_frames <= 0
            or self.event_window_ms <= 0
        ):
            msg = "history_frames must be >=2 and event_window_ms positive."
            raise ValueError(msg)
        if any(horizon < self.event_window_ms for horizon in self.prediction_horizons_ms):
            msg = "Future event targets must not overlap the causal context window."
            raise ValueError(msg)
        if tuple(sorted(set(self.prediction_horizons_ms))) != self.prediction_horizons_ms:
            msg = "prediction_horizons_ms must be unique and increasing."
            raise ValueError(msg)
        if min(
            self.width,
            self.height,
            self.event_bins,
            self.shard_size,
            self.rgb_width,
            self.rgb_height,
            self.garl_time_surface_planes,
        ) <= 0:
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
    segmentation_event_xy: tuple[tuple[float, float], ...] | None = None

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
    pixels = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)
    event_pixels = _event_points(
        pixels,
        source_width=source_width,
        source_height=source_height,
        depth_m=depth_m,
        calibration=calibration,
    )
    if event_pixels is not None:
        scaled = (
            float(event_pixels[:, 0].min()),
            float(event_pixels[:, 1].min()),
            float(event_pixels[:, 0].max()),
            float(event_pixels[:, 1].max()),
        )
    else:
        scaled = (0.0, 0.0, 0.0, 0.0)
    clipped = (
        float(np.clip(scaled[0], 0.0, 1280.0)),
        float(np.clip(scaled[1], 0.0, 720.0)),
        float(np.clip(scaled[2], 0.0, 1280.0)),
        float(np.clip(scaled[3], 0.0, 720.0)),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _event_points(
    pixels: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    depth_m: float | None,
    calibration: _CrossCameraCalibration | None,
) -> np.ndarray | None:
    """Project Blackfly pixels to the event camera at a measured object depth."""

    if calibration is not None and depth_m is not None and depth_m > 0:
        blackfly = calibration.blackfly_intrinsics
        event = calibration.event_intrinsics
        normalized_distorted = np.column_stack(
            ((pixels[:, 0] - blackfly[2]) / blackfly[0], (pixels[:, 1] - blackfly[3]) / blackfly[1])
        )
        normalized = _undistort_radtan(normalized_distorted, calibration.blackfly_distortion)
        points_blackfly = np.column_stack(
            (
                normalized[:, 0] * depth_m,
                normalized[:, 1] * depth_m,
                np.full(pixels.shape[0], depth_m),
                np.ones(pixels.shape[0]),
            )
        )
        points_event = (calibration.event_from_blackfly @ points_blackfly.T).T[:, :3]
        if np.all(points_event[:, 2] > 0):
            event_normalized = points_event[:, :2] / points_event[:, 2:3]
            event_distorted = _distort_radtan(
                event_normalized,
                calibration.event_distortion,
            )
            return np.column_stack(
                (
                    event_distorted[:, 0] * event[0] + event[2],
                    event_distorted[:, 1] * event[1] + event[3],
                )
            )
        return None
    return np.column_stack(
        (
            pixels[:, 0] * 1280.0 / source_width,
            pixels[:, 1] * 720.0 / source_height,
        )
    )


def _normalized_box(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(box, dtype=np.float32) / np.asarray(
        [1280.0, 720.0, 1280.0, 720.0],
        dtype=np.float32,
    )


def _rgb_object_crop(
    handle: h5py.File,
    measurement: LabelMeasurement,
    *,
    config: EvTTCObjectCacheConfig,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Read one official Blackfly frame and retain only its compact object ROI."""

    cached = cache.get(measurement.frame_index)
    if cached is not None:
        return cached
    from PIL import Image

    source = "blackflys/left/data"
    if source not in handle:
        raise ValueError("EvTTC HDF5 lacks blackflys/left/data required for RGB Garl arms.")
    frame = np.asarray(handle[source][measurement.frame_index], dtype=np.uint8)
    x0, y0, x1, y1 = measurement.bbox_xyxy
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    crop_width = max(1.0, (x1 - x0) * config.roi_expansion)
    crop_height = max(1.0, (y1 - y0) * config.roi_expansion)
    image = Image.fromarray(frame)
    crop = image.crop(
        (
            max(0.0, center_x - 0.5 * crop_width),
            max(0.0, center_y - 0.5 * crop_height),
            min(float(image.width), center_x + 0.5 * crop_width),
            min(float(image.height), center_y + 0.5 * crop_height),
        )
    )
    resized = crop.resize((config.rgb_width, config.rgb_height), Image.Resampling.BILINEAR)
    value = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
    cache[measurement.frame_index] = value
    return value


def _rgb_foreground_mask(
    measurement: LabelMeasurement,
    *,
    config: EvTTCObjectCacheConfig,
) -> np.ndarray:
    """Rasterize the labelled object in the same ROI used by the RGB branch."""

    from PIL import Image, ImageDraw

    source_width = measurement.image_width or 1920
    source_height = measurement.image_height or 1200
    image = Image.new("L", (source_width, source_height), color=0)
    draw = ImageDraw.Draw(image)
    if measurement.segmentation_xy and len(measurement.segmentation_xy) >= 3:
        draw.polygon(measurement.segmentation_xy, fill=1)
    else:
        draw.rectangle(measurement.bbox_xyxy, fill=1)
    x0, y0, x1, y1 = measurement.bbox_xyxy
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    crop_width = max(1.0, (x1 - x0) * config.roi_expansion)
    crop_height = max(1.0, (y1 - y0) * config.roi_expansion)
    crop = image.crop(
        (
            max(0.0, center_x - 0.5 * crop_width),
            max(0.0, center_y - 0.5 * crop_height),
            min(float(source_width), center_x + 0.5 * crop_width),
            min(float(source_height), center_y + 0.5 * crop_height),
        )
    )
    resized = crop.resize((256, 256), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _garl_rgb_pair_targets(
    handle: h5py.File,
    history: tuple[_EvTTCState, ...],
    *,
    config: EvTTCObjectCacheConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the two official-style RGB square ROIs and EvTTC height adapter."""

    if len(history) != 3:
        raise ValueError("Official Garl K=1 preprocessing requires exactly three timestamps.")
    from PIL import Image, ImageDraw

    source = "blackflys/left/data"
    if source not in handle:
        raise ValueError("EvTTC HDF5 lacks Blackfly RGB frames required by Garl.")
    endpoints = history[1:]
    boxes = np.asarray(
        [state.measurement.bbox_xyxy for state in endpoints],
        dtype=np.float64,
    )
    common_edge = max(
        1,
        int(
            np.ceil(
                np.maximum(
                    boxes[:, 2] - boxes[:, 0],
                    boxes[:, 3] - boxes[:, 1],
                ).max()
            )
        ),
    )
    rgb_pair: list[np.ndarray] = []
    mask_pair: list[np.ndarray] = []
    visible_heights = _garl_visible_height_targets(history, target_size=config.rgb_height)
    # The released preprocessing concatenates both full RGB frames/masks and
    # then crops them with the square centered on the final endpoint. Event
    # volumes remain centered independently at their corresponding endpoint.
    final_box = boxes[-1]
    center_x = 0.5 * (final_box[0] + final_box[2])
    center_y = 0.5 * (final_box[1] + final_box[3])
    shared_square = (
        int(np.ceil(center_x - 0.5 * common_edge)),
        int(np.ceil(center_y - 0.5 * common_edge)),
        int(np.ceil(center_x + 0.5 * common_edge)),
        int(np.ceil(center_y + 0.5 * common_edge)),
    )
    for state in endpoints:
        measurement = state.measurement
        frame = Image.fromarray(
            np.asarray(handle[source][measurement.frame_index], dtype=np.uint8)
        )
        rgb_pair.append(
            np.asarray(
                frame.crop(shared_square).resize(
                    (config.rgb_width, config.rgb_height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            ).transpose(2, 0, 1)
        )
        mask = Image.new("L", frame.size, color=0)
        draw = ImageDraw.Draw(mask)
        if measurement.segmentation_xy and len(measurement.segmentation_xy) >= 3:
            draw.polygon(measurement.segmentation_xy, fill=1)
        else:
            draw.rectangle(measurement.bbox_xyxy, fill=1)
        mask_pair.append(
            np.asarray(
                mask.crop(shared_square).resize((256, 256), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        )
        # EvTTC has no 3-D object-height target compatible with the original
        # eAP formula.  The declared local adapter supervises the visible 2-D
        # bbox height after the exact shared-square resize.
    return (
        np.stack(rgb_pair),
        np.stack(mask_pair),
        visible_heights,
    )


def _garl_visible_height_targets(
    history: tuple[_EvTTCState, ...],
    *,
    target_size: int,
) -> np.ndarray:
    """Return the declared EvTTC substitute for unavailable 3-D Garl heights."""

    if len(history) != 3:
        raise ValueError("Official Garl K=1 preprocessing requires exactly three timestamps.")
    boxes = np.asarray(
        [state.measurement.bbox_xyxy for state in history[1:]],
        dtype=np.float64,
    )
    common_edge = max(
        1.0,
        float(
            np.maximum(
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ).max()
        ),
    )
    return (
        (boxes[:, 3] - boxes[:, 1]) * float(target_size) / common_edge
    ).astype(np.float32)


def _event_mask(state: _EvTTCState, *, config: EvTTCObjectCacheConfig) -> np.ndarray:
    """Rasterize a projected polygon, with the projected box as an explicit fallback."""

    from PIL import Image, ImageDraw

    image = Image.new("L", (config.width, config.height), color=0)
    draw = ImageDraw.Draw(image)
    if state.segmentation_event_xy and len(state.segmentation_event_xy) >= 3:
        points = [
            (
                float(np.clip(x * config.width / 1280.0, 0.0, config.width - 1.0)),
                float(np.clip(y * config.height / 720.0, 0.0, config.height - 1.0)),
            )
            for x, y in state.segmentation_event_xy
        ]
        draw.polygon(points, fill=1)
    else:
        x0, y0, x1, y1 = state.bbox_event_xyxy
        draw.rectangle(
            (
                x0 * config.width / 1280.0,
                y0 * config.height / 720.0,
                x1 * config.width / 1280.0,
                y1 * config.height / 720.0,
            ),
            fill=1,
        )
    return np.asarray(image, dtype=np.uint8)


def _normalized_event_intrinsics(handle: h5py.File) -> np.ndarray:
    path = "prophesee/event_cam_left/calib/intrinsics"
    if path not in handle:
        return np.asarray((1.0, 1.0, 0.5, 0.5), dtype=np.float32)
    fx, fy, cx, cy = np.asarray(handle[path], dtype=np.float32)
    return np.asarray((fx / 1280.0, fy / 720.0, cx / 1280.0, cy / 720.0), dtype=np.float32)


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


def _voxel_with_metadata(
    handle: h5py.File,
    layout: HDF5EventLayout,
    sequence_id: str,
    *,
    start_us: int,
    end_us: int,
    config: EvTTCObjectCacheConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a voxel plus the two scalar metadata channels used by BASE."""

    events = _read_events(
        handle,
        layout,
        sequence_id,
        start_us=start_us,
        end_us=end_us,
    )
    duration_s = max(events.duration_us * 1e-6, 1e-6)
    metadata = np.asarray(
        (
            np.log1p(events.num_events),
            np.log1p(events.num_events / duration_s),
        ),
        dtype=np.float32,
    )
    voxel = encode_voxel_grid(
        _downsample(events, width=config.width, height=config.height),
        bins=config.event_bins,
        normalize=config.normalize_events,
    ).astype(np.float16)
    return voxel, metadata


def _garl_event_roi(
    handle: h5py.File,
    layout: HDF5EventLayout,
    sequence_id: str,
    history: tuple[_EvTTCState, ...],
    *,
    config: EvTTCObjectCacheConfig,
) -> np.ndarray:
    """Build the official two-interval, 40-plane Garl event representation."""

    if len(history) != 3:
        raise ValueError("Official Garl K=1 preprocessing requires exactly three timestamps.")
    endpoint_boxes = np.asarray(
        [state.bbox_event_xyxy for state in history[1:]],
        dtype=np.float64,
    )
    widths = endpoint_boxes[:, 2] - endpoint_boxes[:, 0]
    heights = endpoint_boxes[:, 3] - endpoint_boxes[:, 1]
    common_edge = max(1, int(np.ceil(np.maximum(widths, heights).max())))
    volumes: list[np.ndarray] = []
    for interval_index, endpoint_index in ((0, 1), (1, 2)):
        events = _read_events(
            handle,
            layout,
            sequence_id,
            start_us=history[interval_index].timestamp_us,
            end_us=history[endpoint_index].timestamp_us,
        )
        box = endpoint_boxes[interval_index]
        center_x = 0.5 * (box[0] + box[2])
        center_y = 0.5 * (box[1] + box[3])
        square = (
            int(np.ceil(center_x - 0.5 * common_edge)),
            int(np.ceil(center_y - 0.5 * common_edge)),
            int(np.ceil(center_x + 0.5 * common_edge)),
            int(np.ceil(center_y + 0.5 * common_edge)),
        )
        volumes.append(
            _garl_time_surface_volume(
                events,
                square,
                planes=config.garl_time_surface_planes,
                target_size=config.rgb_width,
            )
        )
    return np.concatenate(volumes, axis=0).astype(np.float16)


def _garl_time_surface_volume(
    events: EventBatch,
    square_xyxy: tuple[int, int, int, int],
    *,
    planes: int,
    target_size: int,
) -> np.ndarray:
    """Independent implementation of the official Garl time-volume transform."""

    from torch.nn import functional

    x0, y0, x1, y1 = square_xyxy
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    volume = np.zeros((planes, height, width), dtype=np.float32)
    if events.num_events:
        relative_s = (events.t_us - events.t_us[0]).astype(np.float64) * 1e-6
        duration_s = 0.1
        bin_duration = duration_s / planes
        selected = (
            (events.x >= x0)
            & (events.x < x1)
            & (events.y >= y0)
            & (events.y < y1)
            & (relative_s < duration_s - 1e-5)
        )
        x = events.x[selected].astype(np.int64) - x0
        y = events.y[selected].astype(np.int64) - y0
        times = relative_s[selected].astype(np.float32)
        if times.size:
            time_index = np.clip(
                (times.astype(np.float64) / bin_duration).astype(np.int64),
                0,
                planes - 1,
            )
            plane_size = height * width
            flat_index = time_index * plane_size + y * width + x
            order = np.argsort(flat_index, kind="stable")
            sorted_flat = flat_index[order]
            group_start = np.r_[
                0,
                np.flatnonzero(sorted_flat[1:] != sorted_flat[:-1]) + 1,
            ]
            group_count = np.diff(np.r_[group_start, len(sorted_flat)])
            last_position = group_start + group_count - 1
            last_event = order[last_position]
            previous_event = order[np.maximum(last_position - 1, group_start)]
            plane_start = (sorted_flat[last_position] // plane_size) * bin_duration
            previous_time = np.where(
                group_count > 1,
                times[previous_event],
                plane_start,
            )
            values = np.exp(
                -((times[last_event] - previous_time) / bin_duration)
            ).astype(np.float32)
            volume.reshape(-1)[sorted_flat[last_position]] = values
    tensor = torch.from_numpy(volume)[None]
    return (
        functional.interpolate(
            tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=True,
        )[0]
        .numpy()
        .astype(np.float32)
    )


def _actions(
    handle: h5py.File,
    *,
    start_us: int,
    end_us: int,
) -> tuple[np.ndarray, bool]:
    """Return causal event-camera ego velocity, acceleration and yaw rate.

    EvTTC stores velocity in a geographic north/east/up frame and attitude as
    roll/pitch/heading in degrees.  The navigation sensor axes are
    right/forward/up: this is independently confirmed by the documented
    navigation->LiDAR calibration, whose forward axis maps to LiDAR +x.
    Camera-frame velocity is obtained through the complete calibrated chain
    navigation->LiDAR->Blackfly-left->event-left.
    """

    features = np.zeros(8, dtype=np.float32)
    base = "integratedNavigation/data"
    required = [f"{base}/ts", f"{base}/velocity", f"{base}/attitude"]
    if any(name not in handle for name in required):
        return features, False
    navigation_to_event = _navigation_to_event_transform(handle)
    if navigation_to_event is None:
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
    yaw_rad = np.unwrap(np.deg2rad(attitude[:, 2].astype(np.float64)))
    world_from_navigation = np.stack(
        [_world_from_navigation_rfu(yaw) for yaw in yaw_rad],
    )
    event_from_navigation = navigation_to_event[:3, :3]
    navigation_from_event_offset = (
        -event_from_navigation.T @ navigation_to_event[:3, 3]
    )
    event_offset_world = np.einsum(
        "nij,j->ni",
        world_from_navigation,
        navigation_from_event_offset,
    )
    mean_velocity_world = velocity.mean(axis=0, dtype=np.float64)
    if velocity.shape[0] >= 2:
        duration_s = max(float(times[-1] - times[0]) * 1e-6, 1e-6)
        # Add the rigid lever-arm velocity of the event-camera origin. This is
        # the finite displacement of the calibrated camera offset while the
        # vehicle heading changes, not a learned correction.
        mean_velocity_world += (
            event_offset_world[-1] - event_offset_world[0]
        ) / duration_s
    event_from_world_current = event_from_navigation @ world_from_navigation[-1].T
    mean_velocity_event = event_from_world_current @ mean_velocity_world
    features[0] = np.float32(np.linalg.norm(mean_velocity_event))
    features[1:4] = mean_velocity_event.astype(np.float32)
    if velocity.shape[0] >= 2:
        velocity_event = np.stack(
            [
                event_from_navigation @ world_from_navigation[index].T @ value
                for index, value in enumerate(velocity)
            ],
        )
        features[4:7] = (
            (velocity_event[-1] - velocity_event[0]) / duration_s
        ).astype(np.float32)
        # Local UG005 attitude values are Euler angles in degrees (e.g. a
        # heading around 40, not 0.7 radians). Unwrap in radians before the
        # finite difference so crossing 0/360 cannot create a false spike.
        features[7] = np.float32((yaw_rad[-1] - yaw_rad[0]) / duration_s)
    return features, True


def _world_from_navigation_rfu(yaw_rad: float) -> np.ndarray:
    """Return navigation right/forward/up axes in geographic north/east/up."""

    sine = np.sin(yaw_rad)
    cosine = np.cos(yaw_rad)
    return np.asarray(
        (
            (-sine, cosine, 0.0),
            (cosine, sine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ).T


def _navigation_to_event_transform(handle: h5py.File) -> np.ndarray | None:
    """Resolve the documented navigation->LiDAR->RGB->event rigid transform."""

    navigation_to_lidar = "integratedNavigation/data/calib/T_to_lidar"
    lidar_to_blackfly = "livox/lidar/calib/T_to_left_cam"
    event_candidates = (
        "prophesee/event_cam_left/calib/T_bfs_to_prophesee",
        "prophesee/event_cam_left/calib/T_to_left_bfs",
    )
    event_from_blackfly = next(
        (path for path in event_candidates if path in handle),
        None,
    )
    if (
        navigation_to_lidar not in handle
        or lidar_to_blackfly not in handle
        or event_from_blackfly is None
    ):
        return None
    transform = (
        np.asarray(handle[event_from_blackfly], dtype=np.float64)
        @ np.asarray(handle[lidar_to_blackfly], dtype=np.float64)
        @ np.asarray(handle[navigation_to_lidar], dtype=np.float64)
    )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        return None
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        return None
    return transform


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
            segmentation: tuple[tuple[float, float], ...] | None = None
            if measurement.segmentation_xy:
                source_width = measurement.image_width or 1280
                source_height = measurement.image_height or 720
                projected = _event_points(
                    np.asarray(measurement.segmentation_xy, dtype=np.float64),
                    source_width=source_width,
                    source_height=source_height,
                    depth_m=depth,
                    calibration=calibration,
                )
                if projected is not None:
                    projected[:, 0] = np.clip(projected[:, 0], 0.0, 1280.0)
                    projected[:, 1] = np.clip(projected[:, 1], 0.0, 720.0)
                    segmentation = tuple((float(x), float(y)) for x, y in projected)
            states.append(
                _EvTTCState(
                    measurement=measurement,
                    bbox_event_xyxy=box,
                    depth_m=depth,
                    segmentation_event_xy=segmentation,
                )
            )
    return sorted(states, key=lambda state: state.timestamp_us)


def _windows(
    states: list[_EvTTCState],
    config: EvTTCObjectCacheConfig,
) -> list[tuple[tuple[_EvTTCState, ...], dict[int, _EvTTCState]]]:
    timestamps = np.asarray([state.timestamp_us for state in states], dtype=np.int64)
    windows: list[tuple[tuple[_EvTTCState, ...], dict[int, _EvTTCState]]] = []
    earliest_offset = (config.history_frames - 1) * config.history_stride_frames
    for index in range(earliest_offset, len(states)):
        history = tuple(
            states[index - offset]
            for offset in range(
                earliest_offset,
                -1,
                -config.history_stride_frames,
            )
        )
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
    rgb_cache: dict[int, np.ndarray],
) -> dict[str, np.ndarray | str]:
    event_path = sequence.resolve("event_hdf5")
    if event_path is None:
        raise ValueError(f"Sequence {sequence.sequence_id} has no event HDF5 path.")
    context_events: list[np.ndarray] = []
    context_actions: list[np.ndarray] = []
    context_action_mask: list[bool] = []
    context_event_metadata: list[np.ndarray] = []
    context_start: list[int] = []
    for state in history:
        start_us = state.timestamp_us - config.event_window_ms * 1000
        if config.include_context_events:
            voxel, metadata = _voxel_with_metadata(
                handle,
                layout,
                sequence.sequence_id,
                start_us=start_us,
                end_us=state.timestamp_us,
                config=config,
            )
            context_events.append(voxel)
            context_event_metadata.append(metadata)
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
            if config.include_future_events:
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
        if config.include_future_events:
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
    sample: dict[str, np.ndarray | str] = {
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
        "context_intrinsics_normalized": _normalized_event_intrinsics(handle),
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
    if config.include_context_events:
        sample["context_events"] = np.stack(context_events)
        sample["context_event_metadata"] = np.stack(context_event_metadata)
    if config.include_future_events:
        sample["future_events"] = np.stack(future_events)
    if config.include_garl_pair:
        sample["garl_event_roi"] = _garl_event_roi(
            handle,
            layout,
            sequence.sequence_id,
            history,
            config=config,
        )
        sample["garl_visible_heights_px"] = _garl_visible_height_targets(
            history,
            target_size=config.rgb_height,
        )
        sample["garl_delta_t_s"] = np.asarray(
            (history[-1].timestamp_us - history[-2].timestamp_us) * 1e-6,
            dtype=np.float32,
        )
        sample["garl_height_target_source"] = (
            "evttc_visible_bbox_height_after_shared_square_roi_adapter"
        )
    if config.include_rgb:
        sample["context_rgb"] = np.stack(
            [
                _rgb_object_crop(
                    handle,
                    state.measurement,
                    config=config,
                    cache=rgb_cache,
                )
                for state in history
            ]
        )
        if config.include_garl_pair:
            garl_rgb, garl_masks, garl_visible_heights = _garl_rgb_pair_targets(
                handle,
                history,
                config=config,
            )
            sample["garl_rgb_pair"] = garl_rgb
            sample["garl_foreground_mask"] = garl_masks
            sample["garl_visible_heights_px"] = garl_visible_heights
            sample["garl_foreground_source"] = (
                "two_endpoint_isat_polygons_or_bbox_fallback_not_sam"
            )
    if config.include_segmentation_masks:
        sample["context_masks"] = np.stack(
            [_event_mask(state, config=config) for state in history]
        )
        sample["mask_source"] = (
            "projected_isat_polygon_or_projected_bbox_fallback"
        )
    return sample


def _materialize_evttc_sequence(
    task: tuple[DatasetSequence, str, str, EvTTCObjectCacheConfig, int | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence, split, output_text, config, max_windows_per_sequence = task
    output = Path(output_text)
    states = _states(sequence)
    windows = _windows(states, config)
    if max_windows_per_sequence is not None and len(windows) > max_windows_per_sequence:
        indices = np.linspace(0, len(windows) - 1, max_windows_per_sequence, dtype=int)
        windows = [windows[int(index)] for index in indices]
    shards: list[dict[str, Any]] = []
    pending: list[dict[str, np.ndarray | str]] = []
    shard_index = 0
    event_path = sequence.resolve("event_hdf5")
    if event_path is None:
        raise ValueError(f"Sequence {sequence.sequence_id} has no event HDF5 path.")
    layout = discover_event_layout(event_path)
    if layout is None:
        raise ValueError(f"Could not discover event layout for {sequence.sequence_id}.")
    with h5py.File(event_path, "r") as handle:
        rgb_cache: dict[int, np.ndarray] = {}
        for history, future in windows:
            pending.append(
                _sample(
                    sequence,
                    handle,
                    layout,
                    split,
                    history,
                    future,
                    config,
                    rgb_cache,
                )
            )
            if len(pending) >= config.shard_size:
                shards.append(
                    _write_shard(
                        output,
                        sequence_id=sequence.sequence_id,
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
                sequence_id=sequence.sequence_id,
                split_name=split,
                shard_index=shard_index,
                samples=pending,
                config=config,  # type: ignore[arg-type]
            )
        )
    return shards, {
        "sequence_id": sequence.sequence_id,
        "split": split,
        "labelled_states": len(states),
        "windows": len(windows),
    }


def materialize_evttc_object_cache(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    sequence_splits: dict[str, str],
    config: EvTTCObjectCacheConfig | None = None,
    max_windows_per_sequence: int | None = None,
    workers: int = 1,
    maximum_cache_gib: float = 8.0,
    minimum_free_gib: float = 100.0,
    allow_full_dataset_cache: bool = False,
) -> dict[str, Any]:
    """Build a bounded EvTTC cache with causal measured ego motion."""

    config = config or EvTTCObjectCacheConfig()
    bounded_windows = assert_bounded_cache_request(
        max_windows_per_sequence,
        allow_full_dataset_cache=allow_full_dataset_cache,
    )
    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    missing = sorted(set(sequence_splits) - set(sequences))
    if missing:
        raise ValueError(f"EvTTC sequence assignments not in manifest: {missing}.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    output = Path(output_dir)
    estimated_samples = len(sequence_splits) * bounded_windows
    dense_frames = (
        (config.history_frames if config.include_context_events else 0)
        + (
            len(config.prediction_horizons_ms)
            if config.include_future_events
            else 0
        )
    )
    estimated_bytes = (
        estimate_dense_voxel_cache_bytes(
            samples=estimated_samples,
            frames_per_sample=dense_frames,
            channels=2 * config.event_bins,
            height=config.height,
            width=config.width,
        )
        if dense_frames
        else 0
    )
    if config.include_rgb:
        estimated_bytes += (
            estimated_samples
            * config.history_frames
            * 3
            * config.rgb_height
            * config.rgb_width
        )
        estimated_bytes += estimated_samples * 256 * 256
        estimated_bytes += (
            estimated_samples
            * 2
            * 3
            * config.rgb_height
            * config.rgb_width
        )
    if config.include_garl_pair:
        estimated_bytes += (
            estimated_samples
            * 2
            * config.garl_time_surface_planes
            * config.rgb_height
            * config.rgb_width
            * 2
        )
    if config.include_context_events:
        estimated_bytes += estimated_samples * config.history_frames * 2 * 4
    if config.include_segmentation_masks:
        estimated_bytes += (
            estimated_samples * config.history_frames * config.height * config.width
        )
    budget = StorageBudget(
        maximum_cache_gib=maximum_cache_gib,
        minimum_free_gib=minimum_free_gib,
    )
    assert_storage_budget(
        output,
        budget=budget,
        planned_write_bytes=estimated_bytes,
    )
    output.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    tasks = [
        (
            sequences[sequence_id],
            split,
            str(output),
            config,
            max_windows_per_sequence,
        )
        for sequence_id, split in sorted(sequence_splits.items())
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            results = list(executor.map(_materialize_evttc_sequence, tasks))
    else:
        results = [_materialize_evttc_sequence(task) for task in tasks]
    for sequence_shards, summary in results:
        shards.extend(sequence_shards)
        summaries.append(summary)
    payload: dict[str, Any] = {
        "format": "evttc_object_event_jepa_cache_v6",
        "manifest": Path(manifest_path).as_posix(),
        "pre_cropped_events": False,
        "config": asdict(config),
        "sequence_splits": sequence_splits,
        "ttc_label_status": "official_evttc_ttc_table_interpolated_at_label_timestamp",
        "ego_action_status": (
            "calibrated_event_camera_frame_velocity_and_acceleration_from_"
            "world_neu_navigation_plus_unwrapped_yaw_rate_rad_s"
        ),
        "base_metadata_status": "causal_log_event_count_and_log_event_rate_scalars",
        "future_teacher_uses_ego_actions": False,
        "bbox_alignment": "depth_assisted_radtan_blackfly_to_prophesee_calibrated_projection",
        "rgb_cache": (
            "blackfly_object_roi_uint8_128x128" if config.include_rgb else "disabled"
        ),
        "garl_pair_cache": (
            "two_rgb_endpoint_rois_plus_two_20_plane_event_time_surfaces_128x128"
            if config.include_garl_pair
            else "disabled"
        ),
        "context_event_cache": (
            "causal_voxels_plus_two_scalar_base_metadata"
            if config.include_context_events
            else "disabled"
        ),
        "future_event_cache": (
            "causal_disjoint_voxels" if config.include_future_events else "disabled"
        ),
        "mask_cache": (
            "projected_isat_polygon_uint8_with_bbox_fallback"
            if config.include_segmentation_masks
            else "disabled"
        ),
        "normalization": "occupied_voxel_noncentred_q95_magnitude",
        "materialization_workers": min(workers, len(tasks)),
        "storage_policy": {
            "full_dataset_voxel_cache": False,
            "maximum_cache_gib": maximum_cache_gib,
            "minimum_free_gib": minimum_free_gib,
            "estimated_uncompressed_bytes": estimated_bytes,
            "actual_size_bytes": sum(int(shard["size_bytes"]) for shard in shards),
        },
        "sequences": summaries,
        "shards": shards,
        "total_samples": sum(int(shard["samples"]) for shard in shards),
        "total_size_bytes": sum(int(shard["size_bytes"]) for shard in shards),
    }
    actual_size = directory_size_bytes(output)
    if actual_size > budget.maximum_cache_bytes:
        raise RuntimeError(
            "Materialized cache exceeded its hard budget despite the conservative preflight."
        )
    write_structured(output / "manifest.json", payload)
    return payload


__all__ = ["EvTTCObjectCacheConfig", "materialize_evttc_object_cache"]
