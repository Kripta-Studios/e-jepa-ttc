"""Leak-free OOF residual stacking utilities for Object Event TTC v4.26."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualCalibration:
    orientation: float
    slope: float
    train_pearson: float


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    features: tuple[str, ...]
    ridge: float


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fit_residual_calibration(
    score: np.ndarray,
    residual: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> ResidualCalibration:
    score = np.asarray(score, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    p = pearson(score, residual)
    orientation = 1.0 if p >= 0.0 else -1.0
    oriented = orientation * score
    denominator = float(np.dot(oriented, oriented)) + float(epsilon)
    slope = max(0.0, float(np.dot(oriented, residual)) / denominator)
    return ResidualCalibration(orientation=orientation, slope=slope, train_pearson=p)


def apply_residual_calibration(
    score: np.ndarray,
    calibration: ResidualCalibration,
) -> np.ndarray:
    return calibration.orientation * calibration.slope * np.asarray(score, dtype=np.float64)


def residual_design_matrix(
    divergence_residual: np.ndarray,
    vertical_residual: np.ndarray,
    features: Iterable[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    mapping = {
        "divergence": np.asarray(divergence_residual, dtype=np.float64),
        "vertical": np.asarray(vertical_residual, dtype=np.float64),
    }
    names = tuple(str(name) for name in features)
    if len(set(names)) != len(names) or any(name not in mapping for name in names):
        raise ValueError("invalid residual features")
    if not names:
        return np.zeros((len(divergence_residual), 0), dtype=np.float64), names
    return np.column_stack([mapping[name] for name in names]), names


def _ridge_objective(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    ridge: float,
) -> float:
    residual = x @ coefficients - y
    return float(np.mean(residual * residual) + ridge * np.sum(coefficients * coefficients))


def nonnegative_ridge_residual(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Solve the <=2D non-negative residual ridge exactly by active sets."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid residual ridge shapes")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    d = x.shape[1]
    if d == 0:
        return np.zeros(0, dtype=np.float64)
    best = np.zeros(d, dtype=np.float64)
    best_obj = _ridge_objective(x, y, best, ridge)
    for size in range(1, d + 1):
        for subset in combinations(range(d), size):
            idx = np.asarray(subset, dtype=np.int64)
            xs = x[:, idx]
            gram = (xs.T @ xs) / max(len(y), 1)
            rhs = (xs.T @ y) / max(len(y), 1)
            matrix = gram + ridge * np.eye(len(idx), dtype=np.float64)
            try:
                coeff = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            if np.any(coeff < -1.0e-10):
                continue
            candidate = np.zeros(d, dtype=np.float64)
            candidate[idx] = np.maximum(coeff, 0.0)
            objective = _ridge_objective(x, y, candidate, ridge)
            if objective < best_obj:
                best_obj = objective
                best = candidate
    return best


def predict_anchored_residual(
    anchor: np.ndarray,
    x: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float64)
    if x.shape[0] != len(anchor):
        raise ValueError("anchor/design row mismatch")
    return anchor + x @ np.asarray(coefficients, dtype=np.float64)


def track_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    minimum_track_samples: int = 8,
    minimum_negative_track_samples: int = 8,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if minimum_track_samples <= 1 or minimum_negative_track_samples <= 0:
        raise ValueError("invalid track thresholds")
    rows = frame.loc[:, ["sequence_id", "track_id", "target_expansion"]].copy()
    rows["prediction"] = np.asarray(prediction, dtype=np.float64)
    records: list[dict[str, float | int | str]] = []
    for (sequence_id, track_id), group in rows.groupby(["sequence_id", "track_id"], sort=True):
        target = group["target_expansion"].to_numpy(dtype=np.float64)
        pred = group["prediction"].to_numpy(dtype=np.float64)
        neg = target < 0.0
        pos = target >= 0.0
        records.append({
            "sequence_id": str(sequence_id),
            "track_id": str(track_id),
            "count": int(len(group)),
            "negative_count": int(np.sum(neg)),
            "positive_count": int(np.sum(pos)),
            "pearson": pearson(pred, target),
            "negative_accuracy": float(np.mean(pred[neg] < 0.0)) if np.any(neg) else float("nan"),
            "positive_accuracy": float(np.mean(pred[pos] >= 0.0)) if np.any(pos) else float("nan"),
        })
    per_track = pd.DataFrame.from_records(records)
    eligible = per_track[per_track["count"] >= minimum_track_samples]
    negative = per_track[per_track["negative_count"] >= minimum_negative_track_samples]
    metrics: dict[str, float | int] = {
        "track_count": int(len(per_track)),
        "eligible_track_count": int(len(eligible)),
        "negative_track_count": int(len(negative)),
        "track_macro_pearson": float(eligible["pearson"].mean()) if len(eligible) else 0.0,
        "minimum_track_pearson": float(eligible["pearson"].min()) if len(eligible) else 0.0,
        "negative_track_macro_accuracy": float(negative["negative_accuracy"].mean()) if len(negative) else 0.0,
        "minimum_negative_track_accuracy": float(negative["negative_accuracy"].min()) if len(negative) else 0.0,
    }
    return metrics, per_track


__all__ = [
    "ResidualCalibration",
    "ResidualSpec",
    "apply_residual_calibration",
    "fit_residual_calibration",
    "nonnegative_ridge_residual",
    "pearson",
    "predict_anchored_residual",
    "residual_design_matrix",
    "track_metrics",
]
