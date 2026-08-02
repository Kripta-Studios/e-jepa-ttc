"""Source-faithful STRTTC linear model and spatio-temporal refinement.

The equations in this module follow the official MATLAB implementation from
``NAIL-HNU/event_aided_ttc``.  Event selection and normal-flow estimation are
kept outside this file so the solver can be unit-tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class STRTTCLinearResult:
    """Robust three-parameter STRTTC initialization."""

    parameters: np.ndarray
    inlier_mask: np.ndarray
    inlier_ratio: float
    residual_rmse: float

    @property
    def inverse_ttc(self) -> float:
        return float(self.parameters[0])

    @property
    def ttc_seconds(self) -> float:
        return float(1.0 / self.inverse_ttc) if self.inverse_ttc > 0.0 else float("inf")


def construct_strttc_system(
    event_txy: np.ndarray,
    reference_time_s: float,
    normal_flow_xy: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the official STRTTC linear system ``A @ theta = b``."""

    events = np.asarray(event_txy, dtype=np.float64)
    flow = np.asarray(normal_flow_xy, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("event_txy must have shape [N,3] with columns t,x,y.")
    if flow.shape != (events.shape[0], 2):
        raise ValueError("normal_flow_xy must have shape [N,2].")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Focal lengths must be positive.")
    tk = events[:, 0] - float(reference_time_s)
    xk = (events[:, 1] - cx) / fx
    yk = (events[:, 2] - cy) / fy
    dx = flow[:, 0] / fx
    dy = flow[:, 1] / fy
    squared_flow = dx * dx + dy * dy
    design = np.column_stack(
        (
            tk * squared_flow - xk * dx - yk * dy,
            -dx,
            -dy,
        )
    )
    target = -squared_flow
    finite = np.isfinite(design).all(axis=1) & np.isfinite(target)
    return design[finite], target[finite]


def robust_linear_strttc(
    design: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 256,
    sample_size: int = 12,
    squared_residual_threshold: float = 1e-3,
    minimum_inlier_fraction: float = 0.4,
    seed: int = 7,
) -> STRTTCLinearResult:
    """Fit STRTTC parameters with deterministic RANSAC and inlier refit."""

    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] != values.shape[0]:
        raise ValueError("design must be [N,3] and target must be [N].")
    if iterations <= 0 or sample_size < 3 or squared_residual_threshold <= 0.0:
        raise ValueError("RANSAC parameters must be positive.")
    if not 0.0 < minimum_inlier_fraction <= 1.0:
        raise ValueError("minimum_inlier_fraction must be in (0,1].")
    if matrix.shape[0] < sample_size:
        raise ValueError("Not enough normal-flow rows for the requested RANSAC sample.")
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = max(2, int(np.ceil(matrix.shape[0] * minimum_inlier_fraction)) - 1)
    best_error = float("inf")
    for _ in range(iterations):
        indices = rng.choice(matrix.shape[0], size=sample_size, replace=False)
        parameters, *_ = np.linalg.lstsq(matrix[indices], values[indices], rcond=None)
        residual = matrix @ parameters - values
        mask = residual * residual < squared_residual_threshold
        count = int(mask.sum())
        error = float(np.mean(residual[mask] ** 2)) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_mask = mask
            best_count = count
            best_error = error
    if best_mask is None:
        raise RuntimeError("STRTTC RANSAC did not find the minimum inlier support.")
    parameters, *_ = np.linalg.lstsq(matrix[best_mask], values[best_mask], rcond=None)
    residual = matrix @ parameters - values
    return STRTTCLinearResult(
        parameters=parameters,
        inlier_mask=best_mask,
        inlier_ratio=float(best_mask.mean()),
        residual_rmse=float(np.sqrt(np.mean(residual[best_mask] ** 2))),
    )


def warp_strttc_events(
    event_xy: np.ndarray,
    event_time_s: np.ndarray,
    reference_time_s: float,
    parameters: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    """Warp events to the STRTTC reference time using the official model."""

    coordinates = np.asarray(event_xy, dtype=np.float64)
    timestamps = np.asarray(event_time_s, dtype=np.float64).reshape(-1)
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("event_xy must have shape [N,2].")
    if timestamps.shape[0] != coordinates.shape[0] or theta.shape != (3,):
        raise ValueError("timestamps must be [N] and parameters must be [3].")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    normalized = np.empty_like(coordinates)
    normalized[:, 0] = (coordinates[:, 0] - cx) / fx
    normalized[:, 1] = (coordinates[:, 1] - cy) / fy
    delta = (timestamps - float(reference_time_s))[:, None]
    warped = (-theta[0] * normalized + theta[1:3][None]) * delta + normalized
    warped[:, 0] = fx * warped[:, 0] + cx
    warped[:, 1] = fy * warped[:, 1] + cy
    return warped


def _bilinear_sample(
    image: np.ndarray,
    coordinates_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    x = coordinates_xy[:, 0]
    y = coordinates_xy[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
    result = np.zeros(coordinates_xy.shape[0], dtype=np.float64)
    if not valid.any():
        return result, valid
    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    wx = xv - x0v
    wy = yv - y0v
    result[valid] = (
        (1.0 - wx) * (1.0 - wy) * image[y0v, x0v]
        + wx * (1.0 - wy) * image[y0v, x0v + 1]
        + (1.0 - wx) * wy * image[y0v + 1, x0v]
        + wx * wy * image[y0v + 1, x0v + 1]
    )
    return result, valid


def refine_strttc_on_time_surface(
    initial_parameters: np.ndarray,
    event_txy: np.ndarray,
    reference_time_s: float,
    nearest_linear_time_surface: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    robust_c: float = 0.04,
    maximum_function_evaluations: int = 40,
) -> np.ndarray:
    """Refine the linear solution by robust time-surface registration."""

    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError(
            "STRTTC nonlinear refinement requires the optional scipy dependency."
        ) from error

    events = np.asarray(event_txy, dtype=np.float64)
    surface = np.asarray(nearest_linear_time_surface, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 3 or surface.ndim != 2:
        raise ValueError("events must be [N,3] and time surface must be [H,W].")
    if robust_c <= 0.0 or maximum_function_evaluations <= 0:
        raise ValueError("robust_c and maximum_function_evaluations must be positive.")

    fallback = float(np.max(events[:, 0] - reference_time_s))

    def residual(parameters: np.ndarray) -> np.ndarray:
        warped = warp_strttc_events(
            events[:, 1:3],
            events[:, 0],
            reference_time_s,
            parameters,
            intrinsics,
        )
        sampled, valid = _bilinear_sample(surface, warped)
        values = np.full(events.shape[0], fallback, dtype=np.float64)
        values[valid] = sampled[valid]
        robust_scale = np.sqrt(1.0 / (1.0 + values * values / robust_c))
        return values * robust_scale

    result = least_squares(
        residual,
        np.asarray(initial_parameters, dtype=np.float64),
        method="lm",
        max_nfev=maximum_function_evaluations,
    )
    return result.x


def inverse_ttc_at_endpoint(
    inverse_ttc_at_reference: float,
    endpoint_minus_reference_s: float,
) -> float:
    """Transport inverse TTC from the fit reference to a later endpoint."""

    q_reference = float(inverse_ttc_at_reference)
    delta = float(endpoint_minus_reference_s)
    denominator = 1.0 - q_reference * delta
    if q_reference <= 0.0 or denominator <= 1e-8:
        return 0.0
    return q_reference / denominator


__all__ = [
    "STRTTCLinearResult",
    "construct_strttc_system",
    "inverse_ttc_at_endpoint",
    "refine_strttc_on_time_surface",
    "robust_linear_strttc",
    "warp_strttc_events",
]
