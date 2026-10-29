"""Native-event frontend for the official STRTTC geometry.

This is a deterministic Python port of the published MATLAB stages: nearest
linear time surface, contour sampling, local plane normal flow and robust
linear model fitting.  It intentionally consumes raw events rather than the
160x90 neural voxel cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from e_jepa_ttc.geometry.strttc import (
    STRTTCLinearResult,
    construct_strttc_system,
    robust_linear_strttc,
)


@dataclass(frozen=True)
class STRTTCFrontendConfig:
    """Bounded source-port parameters for one raw-event window."""

    maximum_contour_points: int = 512
    spatial_window_size: int = 8
    minimum_neighbour_fraction: float = 0.5
    plane_ransac_iterations: int = 32
    plane_squared_residual_threshold: float = 1e-3
    plane_minimum_inlier_fraction: float = 0.6
    flow_squared_threshold: float = 1e-4
    model_ransac_iterations: int = 256
    model_squared_residual_threshold: float = 1e-3
    model_minimum_inlier_fraction: float = 0.4
    first_gradient_threshold: float = 1e-5
    second_gradient_threshold: float = 1e-3
    seed: int = 7


@dataclass(frozen=True)
class STRTTCFrontendResult:
    """Intermediate source-port products and robust linear estimate."""

    linear: STRTTCLinearResult
    absolute_reference_time_s: float
    reference_time_s: float
    nearest_linear_time_surface: np.ndarray
    contour_txy: np.ndarray
    normal_flow_xy: np.ndarray
    negative_event_support: int


def nearest_linear_time_surface(
    event_txyp: np.ndarray,
    *,
    width: int,
    height: int,
    polarity: int = -1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return per-pixel event time nearest the temporal window midpoint."""

    events = np.asarray(event_txyp, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("event_txyp must have shape [N,4] with t,x,y,p.")
    if width <= 1 or height <= 1 or events.shape[0] == 0:
        raise ValueError("A non-empty event window and positive resolution are required.")
    minimum = float(events[:, 0].min())
    maximum = float(events[:, 0].max())
    reference = 0.5 * (minimum + maximum)
    half_duration = max(0.5 * (maximum - minimum), 1e-8)
    surface = np.full((height, width), half_duration, dtype=np.float64)
    valid = np.zeros((height, width), dtype=np.bool_)
    selected = events[
        (events[:, 3] == polarity)
        & (events[:, 1] >= 0)
        & (events[:, 1] < width)
        & (events[:, 2] >= 0)
        & (events[:, 2] < height)
    ]
    if selected.shape[0] == 0:
        return surface, valid, reference
    relative = selected[:, 0] - reference
    flat = selected[:, 2].astype(np.int64) * width + selected[:, 1].astype(np.int64)
    order = np.lexsort((np.abs(relative), flat))
    ordered_flat = flat[order]
    _, first = np.unique(ordered_flat, return_index=True)
    chosen = order[first]
    y = selected[chosen, 2].astype(np.int64)
    x = selected[chosen, 1].astype(np.int64)
    values = relative[chosen]
    values[values == 0.0] = 1e-10
    surface[y, x] = values
    valid[y, x] = True
    return surface, valid, reference


def _filtered_gradients(surface: np.ndarray) -> tuple[np.ndarray, ...]:
    tensor = torch.from_numpy(surface).float()[None, None]
    padded = functional.pad(tensor, (1, 1, 1, 1), mode="reflect")
    median = padded.unfold(2, 3, 1).unfold(3, 3, 1).median(dim=-1).values.median(dim=-1).values
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = (kernel / kernel.sum())[None, None]
    smooth = functional.conv2d(functional.pad(median, (1, 1, 1, 1), mode="reflect"), kernel)
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[None, None] / 8.0
    sobel_y = sobel_x.transpose(-1, -2)

    def gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded_value = functional.pad(value, (1, 1, 1, 1), mode="reflect")
        return (
            functional.conv2d(padded_value, sobel_x),
            functional.conv2d(padded_value, sobel_y),
        )

    gx, gy = gradient(smooth)
    gxx, gxy = gradient(gx)
    gyx, gyy = gradient(gy)
    return tuple(value[0, 0].numpy() for value in (smooth, gx, gy, gxx, gxy, gyx, gyy))


def select_time_surface_contours(
    surface: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    maximum_points: int,
    first_gradient_threshold: float,
    second_gradient_threshold: float,
    seed: int,
) -> np.ndarray:
    """Select balanced NLTS contour samples as ``[t,x,y]`` rows."""

    filtered, gx, gy, gxx, gxy, gyx, gyy = _filtered_gradients(surface)
    first = gx * gx + gy * gy
    second = gxx * gxx + gxy * gxy + gyx * gyx + gyy * gyy
    y, x = np.nonzero(
        (first > first_gradient_threshold) & (second < second_gradient_threshold) & valid_pixels
    )
    if y.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    per_quadrant = max(1, maximum_points // 4)
    signs = (
        (gx[y, x] < 0) & (gy[y, x] < 0),
        (gx[y, x] > 0) & (gy[y, x] < 0),
        (gx[y, x] > 0) & (gy[y, x] > 0),
        (gx[y, x] < 0) & (gy[y, x] > 0),
    )
    for mask in signs:
        indices = np.flatnonzero(mask)
        if indices.size:
            chosen.append(rng.choice(indices, size=min(per_quadrant, indices.size), replace=False))
    if not chosen:
        return np.empty((0, 3), dtype=np.float64)
    indices = np.concatenate(chosen)[:maximum_points]
    return np.column_stack((filtered[y[indices], x[indices]], x[indices], y[indices]))


def _pixel_event_index(
    events: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = np.flatnonzero(valid)
    flat = y[valid] * width + x[valid]
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    unique, starts, counts = np.unique(sorted_flat, return_index=True, return_counts=True)
    return {
        int(pixel): indices[order[start : start + count]]
        for pixel, start, count in zip(unique, starts, counts, strict=True)
    }


def estimate_plane_normal_flow(
    event_txyp: np.ndarray,
    contour_txy: np.ndarray,
    *,
    width: int,
    height: int,
    config: STRTTCFrontendConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit local event planes and return matched contour rows and normal flow."""

    events = np.asarray(event_txyp, dtype=np.float64)
    contours = np.asarray(contour_txy, dtype=np.float64)
    index = _pixel_event_index(events, width=width, height=height)
    half = config.spatial_window_size // 2
    minimum_events = int(round(config.spatial_window_size**2 * config.minimum_neighbour_fraction))
    accepted_points: list[np.ndarray] = []
    accepted_flow: list[np.ndarray] = []
    for point_index, point in enumerate(contours):
        center_x = int(round(point[1]))
        center_y = int(round(point[2]))
        neighbourhood = [
            index[y * width + x]
            for y in range(max(0, center_y - half), min(height, center_y + half + 1))
            for x in range(max(0, center_x - half), min(width, center_x + half + 1))
            if y * width + x in index
        ]
        if not neighbourhood:
            continue
        selected = events[np.concatenate(neighbourhood)]
        if selected.shape[0] <= minimum_events:
            continue
        local = selected[:, :3] - np.array(
            [selected[0, 0], center_x, center_y],
            dtype=np.float64,
        )
        design = np.column_stack((local[:, 1], local[:, 2], np.ones(local.shape[0])))
        try:
            fitted = robust_linear_strttc(
                design,
                local[:, 0],
                iterations=config.plane_ransac_iterations,
                sample_size=min(12, selected.shape[0]),
                squared_residual_threshold=config.plane_squared_residual_threshold,
                minimum_inlier_fraction=config.plane_minimum_inlier_fraction,
                seed=config.seed + point_index,
            )
        except (RuntimeError, ValueError):
            continue
        a, b = fitted.parameters[:2]
        squared = float(a * a + b * b)
        if squared < config.flow_squared_threshold:
            continue
        accepted_points.append(point)
        accepted_flow.append(np.array([a, b]) / squared)
    if not accepted_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    return np.stack(accepted_points), np.stack(accepted_flow)


def run_strttc_linear_frontend(
    event_txyp: np.ndarray,
    *,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
    config: STRTTCFrontendConfig | None = None,
) -> STRTTCFrontendResult:
    """Run the native NLTS -> normal-flow -> robust STRTTC linear pipeline."""

    resolved = config or STRTTCFrontendConfig()
    surface, valid, absolute_reference = nearest_linear_time_surface(
        event_txyp,
        width=width,
        height=height,
    )
    contours = select_time_surface_contours(
        surface,
        valid,
        maximum_points=resolved.maximum_contour_points,
        first_gradient_threshold=resolved.first_gradient_threshold,
        second_gradient_threshold=resolved.second_gradient_threshold,
        seed=resolved.seed,
    )
    points, normal_flow = estimate_plane_normal_flow(
        event_txyp,
        contours,
        width=width,
        height=height,
        config=resolved,
    )
    if points.shape[0] < 12:
        raise RuntimeError("Fewer than 12 valid STRTTC normal-flow measurements.")
    design, target = construct_strttc_system(
        points,
        0.0,
        normal_flow,
        intrinsics,
    )
    linear = robust_linear_strttc(
        design,
        target,
        iterations=resolved.model_ransac_iterations,
        squared_residual_threshold=resolved.model_squared_residual_threshold,
        minimum_inlier_fraction=resolved.model_minimum_inlier_fraction,
        seed=resolved.seed,
    )
    return STRTTCFrontendResult(
        linear=linear,
        absolute_reference_time_s=absolute_reference,
        reference_time_s=0.0,
        nearest_linear_time_surface=surface,
        contour_txy=points,
        normal_flow_xy=normal_flow,
        negative_event_support=int(valid.sum()),
    )


__all__ = [
    "STRTTCFrontendConfig",
    "STRTTCFrontendResult",
    "estimate_plane_normal_flow",
    "nearest_linear_time_surface",
    "run_strttc_linear_frontend",
    "select_time_surface_contours",
]
