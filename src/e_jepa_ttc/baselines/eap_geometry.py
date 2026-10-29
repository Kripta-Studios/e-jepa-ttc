"""Causal object-geometry baselines for reconstructed public eAP TTC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.evaluation.object_ttc import object_ttc_metrics
from e_jepa_ttc.utils.io import write_structured


def height_ratio_ttc(
    context_boxes_xyxy: np.ndarray,
    context_end_us: np.ndarray,
    *,
    minimum_scale_change: float = 1e-4,
) -> np.ndarray:
    """Estimate TTC from the last two causal apparent-height measurements."""

    boxes = np.asarray(context_boxes_xyxy, dtype=np.float64)
    timestamps = np.asarray(context_end_us, dtype=np.int64)
    if boxes.ndim != 4 or boxes.shape[-1] != 4:
        msg = "context_boxes_xyxy must have shape [N,T,O,4]."
        raise ValueError(msg)
    if timestamps.shape != boxes.shape[:2] or boxes.shape[1] < 2:
        msg = "Context timestamps must have shape [N,T] with T >= 2."
        raise ValueError(msg)
    previous_height = boxes[:, -2, :, 3] - boxes[:, -2, :, 1]
    current_height = boxes[:, -1, :, 3] - boxes[:, -1, :, 1]
    delta_t = (timestamps[:, -1] - timestamps[:, -2]).astype(np.float64) * 1e-6
    ratio = previous_height / np.maximum(current_height, 1e-8)
    denominator = 1.0 - ratio
    estimate = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(denominator)
        & (np.abs(denominator) >= minimum_scale_change)
        & (previous_height > 0)
        & (current_height > 0)
        & (delta_t[:, None] > 0)
    )
    estimate[valid] = np.broadcast_to(delta_t[:, None], estimate.shape)[valid] / denominator[valid]
    return estimate


def depth_velocity_ttc(
    context_depth_m: np.ndarray,
    context_end_us: np.ndarray,
    *,
    minimum_speed_mps: float = 1e-3,
) -> np.ndarray:
    """Estimate signed TTC by a per-window local linear depth fit."""

    depth = np.asarray(context_depth_m, dtype=np.float64)
    timestamps = np.asarray(context_end_us, dtype=np.int64)
    if depth.ndim != 3 or timestamps.shape != depth.shape[:2] or depth.shape[1] < 2:
        msg = "Depth and timestamps must have shapes [N,T,O] and [N,T] with T >= 2."
        raise ValueError(msg)
    estimate = np.full((depth.shape[0], depth.shape[2]), np.nan, dtype=np.float64)
    for sample in range(depth.shape[0]):
        time_s = (timestamps[sample] - timestamps[sample, -1]).astype(np.float64) * 1e-6
        for object_index in range(depth.shape[2]):
            values = depth[sample, :, object_index]
            valid = np.isfinite(values) & np.isfinite(time_s)
            if np.count_nonzero(valid) < 2:
                continue
            slope, _ = np.polyfit(time_s[valid], values[valid], 1)
            if abs(slope) >= minimum_speed_mps and values[-1] > 0:
                estimate[sample, object_index] = -values[-1] / slope
    return estimate


def inverse_ttc_geometry_fusion(
    height_ttc_s: np.ndarray,
    depth_ttc_s: np.ndarray,
) -> np.ndarray:
    """Fuse two geometry estimates conservatively in inverse-TTC space."""

    height = np.asarray(height_ttc_s, dtype=np.float64)
    depth = np.asarray(depth_ttc_s, dtype=np.float64)
    if height.shape != depth.shape:
        msg = "Geometry TTC estimates must have matching shapes."
        raise ValueError(msg)
    inverse = np.stack(
        (
            np.where(np.isfinite(height) & (np.abs(height) >= 0.1), 1.0 / height, np.nan),
            np.where(np.isfinite(depth) & (np.abs(depth) >= 0.1), 1.0 / depth, np.nan),
        ),
        axis=0,
    )
    valid_count = np.sum(np.isfinite(inverse), axis=0)
    fused_inverse = np.divide(
        np.nansum(inverse, axis=0),
        valid_count,
        out=np.full(height.shape, np.nan),
        where=valid_count > 0,
    )
    return np.divide(
        1.0,
        fused_inverse,
        out=np.full(height.shape, np.nan),
        where=np.abs(fused_inverse) >= 1e-6,
    )


def evaluate_eap_geometry_baselines(
    *,
    cache_manifest_path: str | Path,
    splits: tuple[str, ...] = ("validation",),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate causal height, depth and fused geometry on selected cache splits."""

    dataset = EAPObjectCacheDataset(cache_manifest_path, splits=splits)
    boxes: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    sequences: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        boxes.append(sample["context_boxes"].numpy())
        depth.append(sample["context_depth_history_m"].numpy())
        timestamps.append(sample["context_window_end_us"].numpy())
        truth.append(sample["ttc_s"].numpy())
        sequences.append(str(sample["sequence_id"]))
    dataset.close()
    box_array = np.stack(boxes)
    depth_array = np.stack(depth)
    timestamp_array = np.stack(timestamps)
    target = np.stack(truth).reshape(-1)
    predictions = {
        "height_ratio": height_ratio_ttc(box_array, timestamp_array).reshape(-1),
        "depth_velocity": depth_velocity_ttc(depth_array, timestamp_array).reshape(-1),
    }
    predictions["inverse_ttc_fusion"] = inverse_ttc_geometry_fusion(
        predictions["height_ratio"],
        predictions["depth_velocity"],
    )
    payload: dict[str, Any] = {
        "method": "causal_object_geometry_baselines",
        "cache_manifest": str(cache_manifest_path),
        "splits": list(splits),
        "sample_count": int(target.size),
        "sequence_count": len(set(sequences)),
        "ttc_label_status": "reconstructed_public_3d_tracks_not_official_garlttc",
        "metrics": {
            name: object_ttc_metrics(target, prediction) for name, prediction in predictions.items()
        },
    }
    if output_path is not None:
        write_structured(output_path, payload)
    return payload


__all__ = [
    "depth_velocity_ttc",
    "evaluate_eap_geometry_baselines",
    "height_ratio_ttc",
    "inverse_ttc_geometry_fusion",
]
