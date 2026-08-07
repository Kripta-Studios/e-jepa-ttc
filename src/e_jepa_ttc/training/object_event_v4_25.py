"""Constrained train-only readout utilities for object-event v4.25."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GeometryCalibration:
    orientation: float
    slope: float
    train_pearson: float


@dataclass(frozen=True)
class RidgeSpec:
    name: str
    features: tuple[str, ...]
    ridge: float


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fit_geometry_calibration(score: np.ndarray, target: np.ndarray, *, epsilon: float = 1e-12) -> GeometryCalibration:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p = pearson(score, target)
    orientation = 1.0 if p >= 0.0 else -1.0
    oriented = orientation * score
    denominator = float(np.dot(oriented, oriented)) + float(epsilon)
    slope = max(0.0, float(np.dot(oriented, target)) / denominator)
    return GeometryCalibration(orientation=orientation, slope=slope, train_pearson=p)


def apply_geometry_calibration(score: np.ndarray, calibration: GeometryCalibration) -> np.ndarray:
    return calibration.orientation * calibration.slope * np.asarray(score, dtype=np.float64)


def _ridge_objective(x: np.ndarray, y: np.ndarray, coefficients: np.ndarray, prior: np.ndarray, ridge: float) -> float:
    residual = x @ coefficients - y
    return float(np.mean(residual * residual) + ridge * np.sum((coefficients - prior) ** 2))


def nonnegative_ridge_with_prior(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    prior: np.ndarray,
) -> np.ndarray:
    """Solve tiny non-negative ridge exactly by enumerating active sets.

    The feature count in v4.25 is <=3, so active-set enumeration is deterministic,
    dependency-free and easier to audit than a generic optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[1] != len(prior):
        raise ValueError("invalid ridge shapes")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    d = x.shape[1]
    best = np.zeros(d, dtype=np.float64)
    best_obj = _ridge_objective(x, y, best, prior, ridge)
    for size in range(1, d + 1):
        for subset in combinations(range(d), size):
            idx = np.asarray(subset, dtype=np.int64)
            xs = x[:, idx]
            ps = prior[idx]
            gram = (xs.T @ xs) / max(len(y), 1)
            rhs = (xs.T @ y) / max(len(y), 1)
            matrix = gram + ridge * np.eye(len(idx), dtype=np.float64)
            rhs = rhs + ridge * ps
            try:
                coeff = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            if np.any(coeff < -1e-10):
                continue
            candidate = np.zeros(d, dtype=np.float64)
            candidate[idx] = np.maximum(coeff, 0.0)
            obj = _ridge_objective(x, y, candidate, prior, ridge)
            if obj < best_obj:
                best_obj = obj
                best = candidate
    return best


def design_matrix(
    baseline: np.ndarray,
    divergence_expansion: np.ndarray,
    vertical_expansion: np.ndarray,
    features: Iterable[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    mapping = {
        "baseline": np.asarray(baseline, dtype=np.float64),
        "divergence": np.asarray(divergence_expansion, dtype=np.float64),
        "vertical": np.asarray(vertical_expansion, dtype=np.float64),
    }
    names = tuple(str(name) for name in features)
    if not names or "baseline" not in names:
        raise ValueError("every v4.25 readout must contain the baseline anchor")
    if len(set(names)) != len(names) or any(name not in mapping for name in names):
        raise ValueError("invalid readout features")
    x = np.column_stack([mapping[name] for name in names])
    prior = np.asarray([1.0 if name == "baseline" else 0.0 for name in names], dtype=np.float64)
    return x, names, prior


def predict_readout(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(coefficients, dtype=np.float64)


__all__ = [
    "GeometryCalibration",
    "RidgeSpec",
    "apply_geometry_calibration",
    "design_matrix",
    "fit_geometry_calibration",
    "nonnegative_ridge_with_prior",
    "pearson",
    "predict_readout",
]
