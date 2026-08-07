"""Train-only calibration helpers for the v4.19 correspondence probe."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreCalibration:
    orientation: float
    scale: float
    train_pearson: float


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fit_score_calibration(
    score: np.ndarray,
    target_expansion: np.ndarray,
    *,
    minimum_scale: float,
) -> ScoreCalibration:
    """Fit only sign convention and robust scale on train; never center scores."""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target_expansion, dtype=np.float64)
    p = pearson(score, target)
    orientation = 1.0 if p >= 0.0 else -1.0
    oriented = orientation * score
    scale = max(float(np.median(np.abs(oriented))), float(minimum_scale))
    return ScoreCalibration(orientation=orientation, scale=scale, train_pearson=p)


def apply_score_calibration(score: np.ndarray, calibration: ScoreCalibration) -> np.ndarray:
    return calibration.orientation * np.asarray(score, dtype=np.float64) / calibration.scale


def equal_physics_consensus(divergence: np.ndarray, radial: np.ndarray) -> np.ndarray:
    divergence = np.asarray(divergence, dtype=np.float64)
    radial = np.asarray(radial, dtype=np.float64)
    if divergence.shape != radial.shape:
        raise ValueError("physics scores must align")
    return 0.5 * (divergence + radial)


def prediction_from_score(score: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    magnitude = np.abs(np.asarray(magnitude, dtype=np.float64))
    sign = np.where(score < 0.0, -1.0, 1.0)
    return sign * magnitude


__all__ = [
    "ScoreCalibration",
    "apply_score_calibration",
    "equal_physics_consensus",
    "fit_score_calibration",
    "pearson",
    "prediction_from_score",
]
