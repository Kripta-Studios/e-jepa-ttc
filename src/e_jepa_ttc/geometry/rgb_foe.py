"""Source-traceable RGB focus-of-expansion and affine-divergence TTC.

The implementation follows Stabinger et al. (WACV 2016): dense optical flow is
locally approximated by a six-parameter affine field and TTC is obtained from
``div(flow) = 2 / TTC``.  Flow returned by Farneback is displacement between
two frames, therefore the physically dimensioned estimate is
``TTC = 2 * delta_t / divergence_per_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AffineFlowFit:
    """Robust six-parameter affine optical-flow fit."""

    coefficients: np.ndarray
    divergence_per_frame: float
    residual_rmse_px: float
    inlier_fraction: float
    condition_number: float
    sample_count: int


@dataclass(frozen=True)
class RGBFOETTCResult:
    """TTC estimate and diagnostics for one RGB frame pair."""

    ttc_seconds: float
    valid: bool
    reason: str
    divergence_per_second: float
    foe_xy: tuple[float, float] | None
    fit: AffineFlowFit | None


def _validate_flow_inputs(
    points_xy: np.ndarray,
    flow_xy: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    flow = np.asarray(flow_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2].")
    if flow.shape != points.shape:
        raise ValueError("flow_xy must match points_xy.")
    if weights is None:
        base_weights = np.ones(points.shape[0], dtype=np.float64)
    else:
        base_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if base_weights.shape[0] != points.shape[0]:
            raise ValueError("weights must contain one value per flow vector.")
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(flow).all(axis=1)
        & np.isfinite(base_weights)
        & (base_weights > 0.0)
    )
    return points[finite], flow[finite], base_weights[finite]


def fit_affine_flow(
    points_xy: np.ndarray,
    flow_xy: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    robust_iterations: int = 4,
    huber_delta: float = 2.5,
    minimum_points: int = 12,
) -> AffineFlowFit:
    """Fit ``u=c1+c2*x+c3*y, v=c4+c5*x+c6*y`` with Huber IRLS."""

    if robust_iterations < 0:
        raise ValueError("robust_iterations must be non-negative.")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    if minimum_points < 3:
        raise ValueError("minimum_points must be at least three.")
    points, flow, base_weights = _validate_flow_inputs(points_xy, flow_xy, weights)
    if points.shape[0] < minimum_points:
        raise ValueError(
            f"Affine flow requires at least {minimum_points} valid vectors; "
            f"received {points.shape[0]}."
        )

    # Center and scale coordinates for conditioning.  The affine derivatives
    # are transformed back to pixel coordinates below.
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1.0)
    normalized = (points - center) / scale
    one = np.ones(points.shape[0], dtype=np.float64)
    zeros = np.zeros(points.shape[0], dtype=np.float64)
    design = np.block(
        [
            [
                one[:, None],
                normalized[:, :1],
                normalized[:, 1:2],
                zeros[:, None],
                zeros[:, None],
                zeros[:, None],
            ],
            [
                zeros[:, None],
                zeros[:, None],
                zeros[:, None],
                one[:, None],
                normalized[:, :1],
                normalized[:, 1:2],
            ],
        ]
    )
    target = np.concatenate((flow[:, 0], flow[:, 1]))
    vector_weights = np.concatenate((base_weights, base_weights))
    coefficients_normalized = np.zeros(6, dtype=np.float64)
    vector_residual = np.zeros_like(target)

    for _ in range(robust_iterations + 1):
        root = np.sqrt(np.maximum(vector_weights, 1e-12))
        weighted_design = design * root[:, None]
        weighted_target = target * root
        coefficients_normalized, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        vector_residual = design @ coefficients_normalized - target
        paired_residual = np.hypot(
            vector_residual[: points.shape[0]],
            vector_residual[points.shape[0] :],
        )
        robust_scale = 1.4826 * np.median(np.abs(paired_residual - np.median(paired_residual)))
        robust_scale = max(float(robust_scale), 1e-6)
        normalized_residual = paired_residual / robust_scale
        robust = np.ones_like(normalized_residual)
        outside = normalized_residual > huber_delta
        robust[outside] = huber_delta / normalized_residual[outside]
        vector_weights = np.concatenate((base_weights * robust, base_weights * robust))

    c1n, c2n, c3n, c4n, c5n, c6n = coefficients_normalized
    c2 = c2n / scale[0]
    c3 = c3n / scale[1]
    c5 = c5n / scale[0]
    c6 = c6n / scale[1]
    c1 = c1n - c2 * center[0] - c3 * center[1]
    c4 = c4n - c5 * center[0] - c6 * center[1]
    coefficients = np.asarray((c1, c2, c3, c4, c5, c6), dtype=np.float64)
    fitted = np.column_stack(
        (
            c1 + c2 * points[:, 0] + c3 * points[:, 1],
            c4 + c5 * points[:, 0] + c6 * points[:, 1],
        )
    )
    residual = np.linalg.norm(fitted - flow, axis=1)
    robust_scale = max(
        float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
        1e-6,
    )
    inliers = residual <= huber_delta * robust_scale
    gram = design.T @ (vector_weights[:, None] * design)
    return AffineFlowFit(
        coefficients=coefficients,
        divergence_per_frame=float(c2 + c6),
        residual_rmse_px=float(np.sqrt(np.mean(np.square(residual)))),
        inlier_fraction=float(np.mean(inliers)),
        condition_number=float(np.linalg.cond(gram)),
        sample_count=int(points.shape[0]),
    )


def affine_foe_xy(coefficients: np.ndarray) -> tuple[float, float] | None:
    """Return the zero-flow point of an affine field, when numerically stable."""

    c1, c2, c3, c4, c5, c6 = np.asarray(coefficients, dtype=np.float64).reshape(6)
    linear = np.asarray(((c2, c3), (c5, c6)), dtype=np.float64)
    if not np.isfinite(linear).all() or np.linalg.cond(linear) > 1e8:
        return None
    foe = np.linalg.solve(linear, -np.asarray((c1, c4), dtype=np.float64))
    if not np.isfinite(foe).all():
        return None
    return float(foe[0]), float(foe[1])


def ttc_from_affine_fit(
    fit: AffineFlowFit,
    *,
    delta_t_s: float,
    minimum_divergence_per_second: float = 1e-4,
    maximum_ttc_s: float = 12.0,
) -> RGBFOETTCResult:
    """Convert affine divergence into a positive TTC estimate."""

    if delta_t_s <= 0.0 or not np.isfinite(delta_t_s):
        raise ValueError("delta_t_s must be finite and positive.")
    if minimum_divergence_per_second <= 0.0:
        raise ValueError("minimum_divergence_per_second must be positive.")
    divergence_per_second = fit.divergence_per_frame / delta_t_s
    if not np.isfinite(divergence_per_second):
        return RGBFOETTCResult(
            ttc_seconds=float("nan"),
            valid=False,
            reason="non_finite_divergence",
            divergence_per_second=float(divergence_per_second),
            foe_xy=None,
            fit=fit,
        )
    if divergence_per_second <= minimum_divergence_per_second:
        return RGBFOETTCResult(
            ttc_seconds=float("nan"),
            valid=False,
            reason="non_approaching_or_zero_divergence",
            divergence_per_second=float(divergence_per_second),
            foe_xy=affine_foe_xy(fit.coefficients),
            fit=fit,
        )
    ttc = 2.0 / divergence_per_second
    if not np.isfinite(ttc) or ttc <= 0.0:
        return RGBFOETTCResult(
            ttc_seconds=float("nan"),
            valid=False,
            reason="invalid_ttc",
            divergence_per_second=float(divergence_per_second),
            foe_xy=affine_foe_xy(fit.coefficients),
            fit=fit,
        )
    return RGBFOETTCResult(
        ttc_seconds=float(min(ttc, maximum_ttc_s)),
        valid=True,
        reason="ok" if ttc <= maximum_ttc_s else "clipped_high_ttc",
        divergence_per_second=float(divergence_per_second),
        foe_xy=affine_foe_xy(fit.coefficients),
        fit=fit,
    )


def _gray_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError("RGB image must have shape [H,W,C] or [C,H,W].")
    if value.shape[0] in {1, 3, 4} and value.shape[-1] not in {1, 3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] == 1:
        gray = value[..., 0].astype(np.float64)
    elif value.shape[-1] >= 3:
        rgb = value[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("RGB image must have one, three or four channels.")
    if gray.max(initial=0.0) <= 1.0:
        gray = gray * 255.0
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def farneback_affine_ttc(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    delta_t_s: float,
    mask: np.ndarray | None = None,
    grid_step: int = 2,
    minimum_flow_px: float = 0.05,
    maximum_flow_px: float = 64.0,
    farneback_kwargs: dict[str, Any] | None = None,
    maximum_ttc_s: float = 12.0,
) -> RGBFOETTCResult:
    """Estimate RGB TTC with Farneback flow and robust affine divergence."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised by optional dependency guard.
        raise RuntimeError(
            "RGB/FoE evaluation requires the optional 'geometry' dependency "
            "(opencv-python-headless)."
        ) from error
    if grid_step <= 0:
        raise ValueError("grid_step must be positive.")
    previous = _gray_u8(previous_rgb)
    current = _gray_u8(current_rgb)
    if previous.shape != current.shape:
        raise ValueError("RGB frame pair must have identical spatial shape.")
    parameters: dict[str, Any] = {
        "pyr_scale": 0.5,
        "levels": 4,
        "winsize": 21,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 1.5,
        "flags": 0,
    }
    if farneback_kwargs:
        parameters.update(farneback_kwargs)
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **parameters)
    height, width = previous.shape
    y, x = np.mgrid[0:height:grid_step, 0:width:grid_step]
    sampled_flow = flow[::grid_step, ::grid_step]
    points = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)
    vectors = sampled_flow.reshape(-1, 2).astype(np.float64)
    magnitude = np.linalg.norm(vectors, axis=1)
    selected = (
        np.isfinite(vectors).all(axis=1)
        & (magnitude >= minimum_flow_px)
        & (magnitude <= maximum_flow_px)
    )
    if mask is not None:
        mask_value = np.asarray(mask)
        if mask_value.shape != previous.shape:
            raise ValueError("mask must match the RGB frame spatial shape.")
        selected &= mask_value[::grid_step, ::grid_step].reshape(-1) > 0
    try:
        fit = fit_affine_flow(points[selected], vectors[selected])
    except ValueError:
        return RGBFOETTCResult(
            ttc_seconds=float("nan"),
            valid=False,
            reason="insufficient_reliable_flow",
            divergence_per_second=float("nan"),
            foe_xy=None,
            fit=None,
        )
    return ttc_from_affine_fit(
        fit,
        delta_t_s=delta_t_s,
        maximum_ttc_s=maximum_ttc_s,
    )


__all__ = [
    "AffineFlowFit",
    "RGBFOETTCResult",
    "affine_foe_xy",
    "farneback_affine_ttc",
    "fit_affine_flow",
    "ttc_from_affine_fit",
]
