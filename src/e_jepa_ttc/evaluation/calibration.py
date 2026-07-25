"""Calibration utilities for probabilistic TTC and collision-risk outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalIntervalCalibrator:
    """Split-conformal multiplier for symmetric model uncertainty intervals."""

    coverage: float
    scale: float
    calibration_count: int
    minimum_support: int
    support_status: str

    def interval(
        self,
        mean: np.ndarray,
        standard_deviation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated lower and upper TTC bounds."""

        prediction = np.asarray(mean, dtype=np.float64)
        uncertainty = np.asarray(standard_deviation, dtype=np.float64)
        if prediction.shape != uncertainty.shape:
            msg = "Conformal mean and standard deviation shapes must match."
            raise ValueError(msg)
        radius = self.scale * np.maximum(uncertainty, 1e-6)
        return prediction - radius, prediction + radius


def fit_conformal_interval(
    y_true: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    *,
    coverage: float = 0.9,
    min_support: int = 10,
) -> ConformalIntervalCalibrator:
    """Fit a finite-sample split-conformal standardized residual quantile."""

    if not 0.0 < coverage < 1.0:
        msg = "coverage must lie strictly between zero and one."
        raise ValueError(msg)
    target = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(mean, dtype=np.float64)
    uncertainty = np.asarray(standard_deviation, dtype=np.float64)
    if not (target.shape == prediction.shape == uncertainty.shape):
        msg = "Conformal target, mean and standard deviation shapes must match."
        raise ValueError(msg)
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(uncertainty)
    valid &= uncertainty > 0
    scores = np.abs(target[valid] - prediction[valid]) / np.maximum(uncertainty[valid], 1e-6)

    is_supported = scores.size >= min_support
    support_status = "supported" if is_supported else "unsupported"

    if not is_supported:
        return ConformalIntervalCalibrator(
            coverage=coverage,
            scale=1.0,
            calibration_count=int(scores.size),
            minimum_support=min_support,
            support_status=support_status,
        )

    rank = min(scores.size, math.ceil((scores.size + 1) * coverage))
    scale = float(np.partition(scores, rank - 1)[rank - 1])
    return ConformalIntervalCalibrator(
        coverage=coverage,
        scale=scale,
        calibration_count=int(scores.size),
        minimum_support=min_support,
        support_status=support_status,
    )


@dataclass(frozen=True)
class TemperatureScaler:
    """Scalar post-hoc logit temperature selected on calibration data."""

    temperature: float
    calibration_count: int
    minimum_support: int
    support_status: str

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature and return sigmoid probabilities."""

        values = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    grid_size: int = 400,
    min_support: int = 10,
) -> TemperatureScaler:
    """Fit a robust scalar temperature using a deterministic logarithmic grid."""

    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if values.shape != target.shape:
        msg = "Temperature-scaling logits and labels must have matching shapes."
        raise ValueError(msg)
    if minimum <= 0 or maximum <= minimum or grid_size < 2:
        msg = "Invalid temperature search range."
        raise ValueError(msg)
    valid = np.isfinite(values) & np.isfinite(target)
    values = values[valid]
    target = target[valid]

    is_supported = values.size >= min_support
    support_status = "supported" if is_supported else "unsupported"

    if not is_supported:
        return TemperatureScaler(
            temperature=1.0,
            calibration_count=int(values.size),
            minimum_support=min_support,
            support_status=support_status,
        )

    if np.any((target < 0) | (target > 1)):
        msg = "Temperature scaling requires finite binary labels."
        raise ValueError(msg)
    temperatures = np.geomspace(minimum, maximum, grid_size)
    losses = []
    for temperature in temperatures:
        scaled = values / temperature
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - target * scaled)))
    best = int(np.argmin(losses))
    return TemperatureScaler(
        temperature=float(temperatures[best]),
        calibration_count=int(values.size),
        minimum_support=min_support,
        support_status=support_status,
    )


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Compute empirical coverage and mean interval width."""

    target = np.asarray(y_true, dtype=np.float64)
    lower_bound = np.asarray(lower, dtype=np.float64)
    upper_bound = np.asarray(upper, dtype=np.float64)
    if not (target.shape == lower_bound.shape == upper_bound.shape):
        msg = "Interval metric arrays must have matching shapes."
        raise ValueError(msg)
    valid = np.isfinite(target) & np.isfinite(lower_bound) & np.isfinite(upper_bound)
    if not np.any(valid):
        msg = "No finite samples for interval metrics."
        raise ValueError(msg)
    return {
        "coverage": float(
            np.mean((target[valid] >= lower_bound[valid]) & (target[valid] <= upper_bound[valid]))
        ),
        "mean_width_s": float(np.mean(upper_bound[valid] - lower_bound[valid])),
    }


__all__ = [
    "ConformalIntervalCalibrator",
    "TemperatureScaler",
    "fit_conformal_interval",
    "fit_temperature_scaler",
    "interval_metrics",
]
