"""Leakage-safe geometry diagnostics for Object Event TTC v4.4.

The v4.3 multiseed result established a stable event-only ranking signal, but one
held-out sequence retained a severe negative-sign bias.  This module extracts a
small analytic looming descriptor from the same common-coordinate event tensor,
fits train-only weighted ridge calibrators, and reports the official eAP MiD
quantity in addition to the repository's expansion diagnostics.

Nothing in this module consumes boxes, observable motion, RGB, sequence labels as
features, or validation targets during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "geometry_proxy",
    "activity_proxy",
    "radius_velocity_01",
    "radius_velocity_12",
    "radius_acceleration",
    "activity_velocity_01",
    "activity_velocity_12",
    "activity_acceleration",
    "centroid_motion_01",
    "centroid_motion_12",
    "centroid_motion_02",
    "anisotropy_mean",
    "anisotropy_delta",
    "log_radius_mean",
    "log_activity_mean",
    "activity_confidence",
)

EAP_RANGES: tuple[tuple[str, float], ...] = (
    ("crucial", 0.5),
    ("small", 0.3),
    ("large", 0.1),
    ("negative", 0.1),
)


def event_geometry_features(events: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    """Extract causal, box-free spatial looming features.

    Parameters
    ----------
    events:
        Tensor shaped ``[B, 3, C, H, W]``.  Channels are collapsed by absolute
        activity, so the implementation does not assume a particular polarity or
        temporal-bin ordering inside the 12-channel representation.

    Returns
    -------
    torch.Tensor
        Tensor shaped ``[B, len(GEOMETRY_FEATURE_NAMES)]``.
    """

    if events.ndim != 5 or events.shape[1] != 3:
        raise ValueError(f"events must be [B,3,C,H,W], got {tuple(events.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    values = events.float().abs().sum(dim=2)
    batch, steps, height, width = values.shape
    if batch < 1 or steps != 3 or min(height, width) < 2:
        raise ValueError(f"invalid event geometry shape: {tuple(events.shape)}")

    y = torch.linspace(-1.0, 1.0, height, device=values.device, dtype=values.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)

    total = values.sum(dim=(-2, -1)).clamp_min(eps)
    weights = values / total[..., None, None]
    center_x = (weights * xx).sum(dim=(-2, -1))
    center_y = (weights * yy).sum(dim=(-2, -1))
    dx = xx - center_x[..., None, None]
    dy = yy - center_y[..., None, None]
    var_x = (weights * dx.square()).sum(dim=(-2, -1))
    var_y = (weights * dy.square()).sum(dim=(-2, -1))
    cov_xy = (weights * dx * dy).sum(dim=(-2, -1))

    radial_variance = (var_x + var_y).clamp_min(eps)
    log_radius = 0.5 * radial_variance.log()
    log_activity = (total / float(height * width)).clamp_min(eps).log()

    discriminant = ((var_x - var_y).square() + 4.0 * cov_xy.square()).clamp_min(0.0).sqrt()
    eigen_max = 0.5 * (var_x + var_y + discriminant)
    eigen_min = 0.5 * (var_x + var_y - discriminant)
    anisotropy = (eigen_max - eigen_min) / (eigen_max + eigen_min + eps)

    radius_01 = log_radius[:, 1] - log_radius[:, 0]
    radius_12 = log_radius[:, 2] - log_radius[:, 1]
    radius_acceleration = radius_12 - radius_01
    activity_01 = log_activity[:, 1] - log_activity[:, 0]
    activity_12 = log_activity[:, 2] - log_activity[:, 1]
    activity_acceleration = activity_12 - activity_01

    centroid_01 = torch.hypot(center_x[:, 1] - center_x[:, 0], center_y[:, 1] - center_y[:, 0])
    centroid_12 = torch.hypot(center_x[:, 2] - center_x[:, 1], center_y[:, 2] - center_y[:, 1])
    centroid_02 = torch.hypot(center_x[:, 2] - center_x[:, 0], center_y[:, 2] - center_y[:, 0])

    # The t0->t2 interval contains two adjacent TTC steps.  Under the usual
    # pinhole scaling relation, per-step eta is approximately exp(-delta_log_r/2),
    # hence g = 1 - eta.  Event count is treated as an area-like secondary cue.
    geometry_proxy = 1.0 - torch.exp(-0.5 * (log_radius[:, 2] - log_radius[:, 0]))
    activity_proxy = 1.0 - torch.exp(-0.25 * (log_activity[:, 2] - log_activity[:, 0]))
    geometry_proxy = geometry_proxy.clamp(-0.24975, 0.24975)
    activity_proxy = activity_proxy.clamp(-0.24975, 0.24975)

    features = torch.stack(
        (
            geometry_proxy,
            activity_proxy,
            radius_01,
            radius_12,
            radius_acceleration,
            activity_01,
            activity_12,
            activity_acceleration,
            centroid_01,
            centroid_12,
            centroid_02,
            anisotropy.mean(dim=1),
            anisotropy[:, 2] - anisotropy[:, 0],
            log_radius.mean(dim=1),
            log_activity.mean(dim=1),
            torch.sigmoid(log_activity.mean(dim=1)),
        ),
        dim=1,
    )
    if features.shape != (batch, len(GEOMETRY_FEATURE_NAMES)):
        raise AssertionError(f"unexpected geometry feature shape: {tuple(features.shape)}")
    return torch.nan_to_num(features)


def pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {target.shape} != {prediction.shape}")
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]
    if target.size < 2 or np.std(target) <= 1.0e-12 or np.std(prediction) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(target, prediction)[0, 1])
    return value if np.isfinite(value) else 0.0


def sequence_sign_weights(sequence_ids: Iterable[str], target: np.ndarray, *, cap: float = 10.0) -> np.ndarray:
    """Balance sequence/sign cells while bounding rare-cell leverage."""

    sequence = np.asarray(list(sequence_ids), dtype=object)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if sequence.shape[0] != target.shape[0]:
        raise ValueError("sequence_ids and target must have the same length")
    if cap < 1.0:
        raise ValueError("cap must be at least 1")
    signs = np.where(target < 0.0, "negative", "positive")
    cells = np.asarray([f"{seq}|{sign}" for seq, sign in zip(sequence, signs, strict=True)], dtype=object)
    unique, counts = np.unique(cells, return_counts=True)
    count_by_cell = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    raw = np.asarray([1.0 / count_by_cell[cell] for cell in cells], dtype=np.float64)
    median = float(np.median(raw))
    if median > 0.0:
        raw = np.minimum(raw, median * cap)
    raw /= max(float(np.mean(raw)), 1.0e-12)
    return raw


@dataclass(frozen=True)
class StandardizedRidge:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("ridge feature dimension mismatch")
        standardized = (values - self.mean) / self.scale
        design = np.column_stack((np.ones(len(values), dtype=np.float64), standardized))
        return design @ self.coefficients


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    alpha: float = 1.0,
) -> StandardizedRidge:
    """Fit a numerically stable ridge model with an unpenalized intercept."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.shape[0] or values.shape[0] < 2:
        raise ValueError("invalid ridge training shapes")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("ridge inputs must be finite")

    if sample_weight is None:
        weight = np.ones(len(target), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weight.shape != target.shape or np.any(weight <= 0.0) or not np.isfinite(weight).all():
            raise ValueError("sample_weight must be finite and positive")
    weight = weight / max(float(np.mean(weight)), 1.0e-12)

    mean = np.average(values, axis=0, weights=weight)
    centered = values - mean
    variance = np.average(np.square(centered), axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 1.0e-12))
    standardized = centered / scale
    design = np.column_stack((np.ones(len(values), dtype=np.float64), standardized))
    root_weight = np.sqrt(weight)[:, None]
    weighted_design = design * root_weight
    weighted_target = target * root_weight[:, 0]
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs
    return StandardizedRidge(mean=mean, scale=scale, coefficients=coefficients, alpha=float(alpha))


def branch_metrics(
    target_expansion: np.ndarray,
    prediction_expansion: np.ndarray,
    delta_t_s: np.ndarray,
    *,
    ttc_clip_seconds: float = 60.0,
    minimum_abs_expansion: float = 1.0e-4,
) -> dict[str, float]:
    target = np.asarray(target_expansion, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_expansion, dtype=np.float64).reshape(-1)
    delta_t = np.asarray(delta_t_s, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.shape != delta_t.shape:
        raise ValueError("branch metric shapes must match")
    positive = target >= 0.0
    negative = target < 0.0
    positive_accuracy = float(np.mean(prediction[positive] >= 0.0)) if positive.any() else 0.0
    negative_accuracy = float(np.mean(prediction[negative] < 0.0)) if negative.any() else 0.0
    sign = np.where(prediction < 0.0, -1.0, 1.0)
    denominator = sign * np.maximum(np.abs(prediction), minimum_abs_expansion)
    predicted_ttc = np.clip(delta_t / denominator, -ttc_clip_seconds, ttc_clip_seconds)
    return {
        "count": int(target.size),
        "negative_count": int(negative.sum()),
        "positive_count": int(positive.sum()),
        "pearson": pearson(target, prediction),
        "expansion_mae": float(np.mean(np.abs(target - prediction))),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_sign_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
        "ttc_saturation_rate": float(
            np.mean(np.abs(predicted_ttc) >= ttc_clip_seconds * (1.0 - 1.0e-6))
        ),
    }


def _range_mask(target_ttc_s: np.ndarray, name: str) -> np.ndarray:
    ttc = np.asarray(target_ttc_s, dtype=np.float64)
    if name == "crucial":
        return (ttc > 0.0) & (ttc <= 3.0)
    if name == "small":
        return (ttc > 3.0) & (ttc <= 6.0)
    if name == "large":
        return (ttc > 6.0) & (ttc <= 10.0)
    if name == "negative":
        return (ttc >= -10.0) & (ttc < 0.0)
    raise KeyError(name)


def official_eap_metrics(
    target_expansion: np.ndarray,
    prediction_expansion: np.ndarray,
    delta_t_s: np.ndarray,
    target_ttc_s: np.ndarray,
    *,
    max_abs_expansion: float = 0.25,
) -> dict[str, object]:
    """Compute eAP MiD and RTE using the paper's four TTC ranges.

    The returned values are directly formula-compatible with the paper, but a
    validation subset is not interchangeable with the official eAP test split.
    """

    target = np.asarray(target_expansion, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_expansion, dtype=np.float64).reshape(-1)
    delta_t = np.asarray(delta_t_s, dtype=np.float64).reshape(-1)
    target_ttc = np.asarray(target_ttc_s, dtype=np.float64).reshape(-1)
    if not (target.shape == prediction.shape == delta_t.shape == target_ttc.shape):
        raise ValueError("official metric shapes must match")
    if not 0.0 < max_abs_expansion < 1.0:
        raise ValueError("max_abs_expansion must lie in (0,1)")

    limit = max_abs_expansion * 0.999
    target = np.clip(target, -limit, limit)
    prediction = np.clip(prediction, -limit, limit)
    target_eta = np.clip(1.0 - target, 1.0e-8, None)
    prediction_eta = np.clip(1.0 - prediction, 1.0e-8, None)
    mid = np.abs(np.log(prediction_eta) - np.log(target_eta)) * 1.0e4

    sign = np.where(prediction < 0.0, -1.0, 1.0)
    denominator = sign * np.maximum(np.abs(prediction), 1.0e-6)
    predicted_ttc = delta_t / denominator
    rte = np.abs((target_ttc - predicted_ttc) / np.maximum(np.abs(target_ttc), 1.0e-8)) * 100.0

    by_range: dict[str, dict[str, float | int | None]] = {}
    weighted_mid = 0.0
    weighted_rte = 0.0
    all_ranges_present = True
    for name, weight in EAP_RANGES:
        mask = _range_mask(target_ttc, name)
        count = int(mask.sum())
        if count == 0:
            all_ranges_present = False
            by_range[name] = {"count": 0, "weight": weight, "mid": None, "rte_percent": None}
            continue
        range_mid = float(np.mean(mid[mask]))
        range_rte = float(np.mean(rte[mask]))
        weighted_mid += weight * range_mid
        weighted_rte += weight * range_rte
        by_range[name] = {
            "count": count,
            "weight": weight,
            "mid": range_mid,
            "rte_percent": range_rte,
        }
    return {
        "mid_mean_unweighted": float(np.mean(mid)),
        "rte_mean_unweighted_percent": float(np.mean(rte)),
        "weighted_mid": float(weighted_mid) if all_ranges_present else None,
        "weighted_rte_percent": float(weighted_rte) if all_ranges_present else None,
        "all_ranges_present": all_ranges_present,
        "by_range": by_range,
    }


__all__ = [
    "EAP_RANGES",
    "GEOMETRY_FEATURE_NAMES",
    "StandardizedRidge",
    "branch_metrics",
    "event_geometry_features",
    "fit_weighted_ridge",
    "official_eap_metrics",
    "pearson",
    "sequence_sign_weights",
]
