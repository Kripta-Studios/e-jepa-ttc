"""Numerically explicit preprocessing compatible with the official Garl release."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn.functional import grid_sample


def official_timevolume_roi_np(
    expand_box: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    *,
    time_window_s: float = 0.1,
    number_of_planes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``garl_ttc.datasets.event_representation`` exactly."""

    if time_window_s <= 0.0 or number_of_planes <= 0:
        raise ValueError("time_window_s and number_of_planes must be positive.")
    timestamps = np.asarray(t_us, dtype=np.int64)
    if timestamps.size == 0:
        width = int(expand_box[2] - expand_box[0])
        height = int(expand_box[3] - expand_box[1])
        return np.zeros((number_of_planes, height, width), dtype=np.float32), np.zeros(
            number_of_planes, dtype=np.int32
        )
    tus = timestamps - timestamps[0]
    ts = tus * 1e-6
    tbin = time_window_s / number_of_planes
    xmin, ymin, xmax, ymax = (int(value) for value in expand_box)
    width = int(xmax - xmin)
    height = int(ymax - ymin)
    if width <= 0 or height <= 0:
        raise ValueError("expand_box must have positive width and height.")
    x_values = np.asarray(x)[...]
    y_values = np.asarray(y)[...]
    mask = (
        (x_values >= xmin)
        & (x_values < xmax)
        & (y_values >= ymin)
        & (y_values < ymax)
        & (ts < time_window_s - 1e-5)
    )
    x_roi = x_values[mask].astype(np.int64) - xmin
    y_roi = y_values[mask].astype(np.int64) - ymin
    ts_roi = ts[mask].astype(np.float32)
    timevolume = np.zeros((number_of_planes, height, width), dtype=np.float32)
    if len(ts_roi) == 0:
        return timevolume, np.zeros(number_of_planes, dtype=np.int32)
    time_ind = (ts_roi.astype(np.float64) / tbin).astype(np.int64)
    evcount = np.bincount(time_ind, minlength=number_of_planes).astype(np.int32)
    plane_size = height * width
    flat_idx = time_ind * plane_size + y_roi * width + x_roi
    order = np.argsort(flat_idx, kind="stable")
    sorted_flat = flat_idx[order]
    group_start = np.r_[0, np.flatnonzero(sorted_flat[1:] != sorted_flat[:-1]) + 1]
    group_count = np.diff(np.r_[group_start, len(sorted_flat)])
    last_pos = group_start + group_count - 1
    last_event = order[last_pos]
    prev_event = order[np.maximum(last_pos - 1, group_start)]
    plane_start = (sorted_flat[last_pos] // plane_size) * tbin
    prev_ts = np.where(group_count > 1, ts_roi[prev_event], plane_start)
    values = np.exp(-((ts_roi[last_event] - prev_ts) / tbin)).astype(np.float32)
    timevolume.reshape(-1)[sorted_flat[last_pos]] = values
    return timevolume, evcount


def official_resize_feature(
    feature: np.ndarray,
    target_size: tuple[int, int],
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Match the release's ``grid_sample`` resize, including its coordinates."""

    if feature.ndim != 3:
        raise ValueError("feature must have shape [C,H,W].")
    target_height, target_width = target_size
    _, image_height, image_width = feature.shape
    image = torch.from_numpy(np.ascontiguousarray(feature))[None].to(device)
    xs = torch.linspace(0, image_width, steps=target_width, dtype=image.dtype, device=image.device)
    ys = torch.linspace(
        0, image_height, steps=target_height, dtype=image.dtype, device=image.device
    )
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack(
        (x_grid / image_width * 2.0 - 1.0, y_grid / image_height * 2.0 - 1.0),
        dim=-1,
    )
    return grid_sample(
        image,
        coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].cpu()


def official_resize_roi(
    feature: np.ndarray,
    square: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Match the release's normalized full-frame RGB ROI sampling."""

    if feature.ndim != 3:
        raise ValueError("feature must have shape [C,H,W].")
    target_height, target_width = target_size
    _, image_height, image_width = feature.shape
    image = torch.from_numpy(np.ascontiguousarray(feature))[None]
    x_min, y_min, x_max, y_max = square
    xs = torch.linspace(x_min, x_max, steps=target_width, dtype=image.dtype)
    ys = torch.linspace(y_min, y_max, steps=target_height, dtype=image.dtype)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack(
        (
            x_grid / image_width * 2.0 - 1.0,
            y_grid / image_height * 2.0 - 1.0,
        ),
        dim=-1,
    )
    return grid_sample(
        image,
        coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].cpu()


def official_square_box(
    boxes: list[tuple[float, float, float, float]], index: int
) -> tuple[int, int, int, int]:
    """Match the release's integer center and ceil-based square construction."""

    max_edge = max(max(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) for box in boxes)
    xmin, ymin, xmax, ymax = (int(value) for value in boxes[index])
    cx, cy = int((xmin + xmax) / 2.0), int((ymin + ymax) / 2.0)
    return (
        int(math.ceil(cx - max_edge / 2.0)),
        int(math.ceil(cy - max_edge / 2.0)),
        int(math.ceil(cx + max_edge / 2.0)),
        int(math.ceil(cy + max_edge / 2.0)),
    )


__all__ = [
    "official_resize_feature",
    "official_resize_roi",
    "official_square_box",
    "official_timevolume_roi_np",
]
