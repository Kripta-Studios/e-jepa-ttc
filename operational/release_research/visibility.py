"""Coordinate and persistent-surface diagnostics; no target-based patch selection."""

from __future__ import annotations

import numpy as np

from .physics import Array


def sensor_log_scale_change(crop_change: Array, scale_a: Array, scale_b: Array) -> Array:
    """Remove known crop zoom from log size change before converting to a rate."""
    if crop_change.shape != scale_a.shape or crop_change.shape != scale_b.shape:
        raise ValueError("Crop change and endpoint scales must align")
    if not all(np.isfinite(a).all() for a in (crop_change, scale_a, scale_b)):
        raise ValueError("Nonfinite coordinates")
    if np.any(scale_a <= 0) or np.any(scale_b <= 0):
        raise ValueError("Crop scales must be positive")
    return crop_change - np.log(scale_b / scale_a)


def sensor_velocity(
    crop_xy: Array,
    crop_velocity: Array,
    scale: float,
    scale_rate: float,
    origin_velocity: Array,
) -> Array:
    """Invert d[s(x-o)]/dt including moving origin and zoom, in sensor pixels/s."""
    if crop_xy.ndim != 2 or crop_xy.shape[1] != 2 or crop_velocity.shape != crop_xy.shape:
        raise ValueError("Expected crop positions/velocities [N,2]")
    if origin_velocity.shape != (2,) or not np.isfinite([scale, scale_rate]).all() or scale <= 0:
        raise ValueError("Invalid crop transform")
    if not all(np.isfinite(a).all() for a in (crop_xy, crop_velocity, origin_velocity)):
        raise ValueError("Nonfinite crop transform")
    return (crop_velocity - (scale_rate / scale) * crop_xy) / scale + origin_velocity


def surface_log_expansion(jacobians: Array) -> tuple[Array, np.ndarray]:
    """Return half log determinant; orientation loss and degeneracy remain invalid."""
    if jacobians.ndim != 3 or jacobians.shape[1:] != (2, 2):
        raise ValueError("Expected Jacobians [N,2,2]")
    if not np.isfinite(jacobians).all():
        raise ValueError("Nonfinite surface Jacobian")
    sign, logdet = np.linalg.slogdet(jacobians)
    valid = (sign > 0) & np.isfinite(logdet)
    return np.where(valid, 0.5 * logdet, np.nan), valid


def composition_residual(
    ab: Array,
    bc_at_ab: Array,
    ac_independent: Array,
    visible: np.ndarray,
    *,
    independent_direct_estimate: bool,
) -> dict[str, float | int | None]:
    """Compare independent direct transport with composed persistent correspondences."""
    if not independent_direct_estimate:
        raise ValueError("Algebraically composed a->c supplies no independent evidence")
    if ab.shape != bc_at_ab.shape or ab.shape != ac_independent.shape or ab.shape != visible.shape:
        raise ValueError("Surface correspondence shapes differ")
    if visible.dtype != np.bool_ or ab.ndim != 1 or len(ab) == 0:
        raise ValueError("Expected nonempty vectors and a boolean visibility mask")
    valid = visible & np.isfinite(ab) & np.isfinite(bc_at_ab) & np.isfinite(ac_independent)
    count = int(valid.sum())
    return {
        "required": len(ab),
        "accepted": count,
        "coverage": count / len(ab),
        "mean_absolute_residual": float(
            np.abs(ac_independent[valid] - ab[valid] - bc_at_ab[valid]).mean()
        )
        if count
        else None,
    }
