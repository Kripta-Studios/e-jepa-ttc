"""Signed TTC diagnostics retaining missingness and clustered comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np


def bbox_ttc(boxes: np.ndarray, timestamps_us: np.ndarray, axis: str = "height") -> float:
    """Two-endpoint constant-speed expansion estimate at the CURRENT timestamp."""
    if len(boxes) < 2 or len(boxes) != len(timestamps_us):
        return float("nan")
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        return float("nan")
    size = {"width": widths, "height": heights, "area": np.sqrt(widths * heights)}[axis]
    dt = (int(timestamps_us[-1]) - int(timestamps_us[-2])) / 1e6
    if dt <= 0 or not np.isfinite(size).all() or size[-1] == size[-2]:
        return float("nan")
    return float(dt / (size[-1] / size[-2] - 1))


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Report conditional error alongside its full denominator and finite coverage."""
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("Metric inputs must be matched vectors")
    valid_target = np.isfinite(target)
    mask = valid_target & np.isfinite(prediction)
    relative = mask & (np.abs(target) > 0)
    error = np.abs(target - prediction)
    return {
        "population": len(target),
        "valid_targets": int(valid_target.sum()),
        "finite": int(mask.sum()),
        "coverage": float(mask.sum() / max(1, valid_target.sum())),
        "mae_seconds": float(error[mask].mean()) if mask.any() else None,
        "rmse_seconds": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
        "relative_rows": int(relative.sum()),
        "relative_absolute_error_pct": (
            float(np.mean(100 * error[relative] / np.abs(target[relative])))
            if relative.any()
            else None
        ),
        "wrong_sign": int(np.sum(np.sign(target[relative]) != np.sign(prediction[relative]))),
    }


def phase_interval_metrics(target_phase: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Score native 80% phase intervals; a zero crossing is a disjoint TTC interval."""
    if not (target_phase.shape == lower.shape == upper.shape):
        raise ValueError("Interval shapes differ")
    if not np.isfinite(np.stack((target_phase, lower, upper))).all() or np.any(lower > upper):
        raise ValueError("Invalid native phase interval")
    return {
        "coverage_80": float(np.mean((target_phase >= lower) & (target_phase <= upper))),
        "mean_width_phase": float(np.mean(upper - lower)),
        "crosses_infinite_ttc_fraction": float(np.mean((lower <= 0) & (upper >= 0))),
        "coordinate": "phase; zero-crossing TTC sets must not be treated as bounded intervals",
    }


def paired_group_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 7,
) -> dict:
    """Resample whole groups, preserving paired errors and within-group membership."""
    if not (baseline.shape == candidate.shape == groups.shape) or baseline.ndim != 1:
        raise ValueError("Bootstrap inputs differ")
    if not np.isfinite(np.stack((baseline, candidate))).all() or repetitions < 1:
        raise ValueError("Bootstrap requires finite paired inputs and positive repetitions")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two groups are required")
    delta = candidate - baseline
    sums = np.array([delta[groups == group].sum() for group in unique])
    counts = np.array([np.sum(groups == group) for group in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), (repetitions, len(unique)))
    sampled = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    return {
        "groups": len(unique),
        "mean_difference": float(delta.mean()),
        "lower_95": float(lo),
        "upper_95": float(hi),
        "seed": seed,
        "repetitions": repetitions,
        "unit": "declared_group_not_individual_windows",
    }
