"""Contrast maximization for raw-event radial expansion TTC.

This is a bounded, source-traceable implementation of the contrast objective
from Gallego et al. (CVPR 2018).  Events are warped to the final timestamp by a
continuous radial-expansion model and accumulated bilinearly into an image of
warped events (IWE).  The selected inverse TTC maximizes IWE variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class CMaxResult:
    """Bounded raw-event contrast-maximization result."""

    inverse_ttc_per_s: float
    ttc_seconds: float
    valid: bool
    reason: str
    contrast: float
    null_contrast: float
    relative_contrast_gain: float
    survival_fraction: float
    confidence: float
    evaluations: int


def warp_radial_events(
    xy: np.ndarray,
    times_s: np.ndarray,
    *,
    inverse_ttc_per_s: float,
    center_xy: tuple[float, float],
    event_centers_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Warp events from their timestamps to the latest time under looming."""

    points = np.asarray(xy, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != times.shape[0]:
        raise ValueError("xy and times_s must have shapes [N,2] and [N].")
    if not np.isfinite(points).all() or not np.isfinite(times).all():
        raise ValueError("CMax input events must be finite.")
    relative = times - float(np.max(times))
    endpoint_center = np.asarray(center_xy, dtype=np.float64)
    if event_centers_xy is None:
        event_centers = np.broadcast_to(endpoint_center, points.shape)
    else:
        event_centers = np.asarray(event_centers_xy, dtype=np.float64)
        if event_centers.shape != points.shape or not np.isfinite(event_centers).all():
            raise ValueError("event_centers_xy must be finite and shape matched with xy.")
    scale = np.exp(-float(inverse_ttc_per_s) * relative)
    return endpoint_center + scale[:, None] * (points - event_centers)


def image_of_warped_events(
    warped_xy: np.ndarray,
    polarities: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Bilinearly accumulate positive/negative events into two IWE channels."""

    height, width = image_shape
    if height <= 1 or width <= 1:
        raise ValueError("image_shape must contain positive dimensions greater than one.")
    points = np.asarray(warped_xy, dtype=np.float64)
    polarity = np.asarray(polarities).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != polarity.shape[0]:
        raise ValueError("warped_xy and polarities must be event-count matched.")
    finite = np.isfinite(points).all(axis=1)
    x = points[:, 0]
    y = points[:, 1]
    inside = finite & (x >= 0.0) & (x < width - 1.0) & (y >= 0.0) & (y < height - 1.0)
    if not np.any(inside):
        return np.zeros((2, height, width), dtype=np.float64), 0.0
    x = x[inside]
    y = y[inside]
    channel = (polarity[inside] > 0).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    dx = x - x0
    dy = y - y0
    image = np.zeros((2, height, width), dtype=np.float64)
    for offset_x, offset_y, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        np.add.at(image, (channel, y0 + offset_y, x0 + offset_x), weight)
    return image, float(np.mean(inside))


def iwe_contrast(image: np.ndarray) -> float:
    """Return mean channel variance, the canonical CMax contrast objective."""

    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("IWE must have shape [2,H,W].")
    return float(np.mean(np.var(value, axis=(1, 2))))


def _contrast_for_rate(
    rate: float,
    *,
    xy: np.ndarray,
    times_s: np.ndarray,
    polarities: np.ndarray,
    image_shape: tuple[int, int],
    center_xy: tuple[float, float],
    event_centers_xy: np.ndarray | None,
    minimum_survival_fraction: float,
) -> tuple[float, float]:
    warped = warp_radial_events(
        xy,
        times_s,
        inverse_ttc_per_s=rate,
        center_xy=center_xy,
        event_centers_xy=event_centers_xy,
    )
    image, survival = image_of_warped_events(
        warped,
        polarities,
        image_shape=image_shape,
    )
    if survival < minimum_survival_fraction:
        return 0.0, survival
    # The survival factor prevents a false optimum that simply warps difficult
    # events outside the image.
    return iwe_contrast(image) * survival * survival, survival


def maximize_radial_event_contrast(
    xy: np.ndarray,
    times_s: np.ndarray,
    polarities: np.ndarray,
    *,
    image_shape: tuple[int, int],
    center_xy: tuple[float, float] | None = None,
    event_centers_xy: np.ndarray | None = None,
    minimum_ttc_s: float = 0.25,
    maximum_ttc_s: float = 12.0,
    coarse_steps: int = 33,
    minimum_events: int = 100,
    minimum_survival_fraction: float = 0.8,
    minimum_relative_contrast_gain: float = 0.01,
) -> CMaxResult:
    """Optimize signed expansion, accepting only a positive collision solution."""

    points = np.asarray(xy, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    polarity = np.asarray(polarities).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must have shape [N,2].")
    if points.shape[0] != times.shape[0] or points.shape[0] != polarity.shape[0]:
        raise ValueError("CMax raw-event fields must have equal length.")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(times)
    points = points[finite]
    times = times[finite]
    polarity = polarity[finite]
    centers = (
        np.asarray(event_centers_xy, dtype=np.float64)[finite]
        if event_centers_xy is not None
        else None
    )
    if centers is not None and (centers.shape != points.shape or not np.isfinite(centers).all()):
        raise ValueError("event_centers_xy must be finite and shape matched with xy.")
    if points.shape[0] < minimum_events:
        return CMaxResult(
            inverse_ttc_per_s=float("nan"),
            ttc_seconds=float("nan"),
            valid=False,
            reason="insufficient_events",
            contrast=float("nan"),
            null_contrast=float("nan"),
            relative_contrast_gain=float("nan"),
            survival_fraction=0.0,
            confidence=0.0,
            evaluations=0,
        )
    if maximum_ttc_s <= minimum_ttc_s or minimum_ttc_s <= 0.0:
        raise ValueError("TTC search bounds are invalid.")
    if coarse_steps < 9:
        raise ValueError("coarse_steps must be at least nine.")
    if center_xy is None:
        center = (0.5 * (image_shape[1] - 1.0), 0.5 * (image_shape[0] - 1.0))
    else:
        center = center_xy
    maximum_rate = 1.0 / minimum_ttc_s
    minimum_positive_rate = 1.0 / maximum_ttc_s
    evaluations = 0

    def objective(rate: float) -> float:
        nonlocal evaluations
        evaluations += 1
        contrast, _ = _contrast_for_rate(
            float(rate),
            xy=points,
            times_s=times,
            polarities=polarity,
            image_shape=image_shape,
            center_xy=center,
            event_centers_xy=centers,
            minimum_survival_fraction=minimum_survival_fraction,
        )
        return contrast

    grid = np.linspace(-maximum_rate, maximum_rate, coarse_steps, dtype=np.float64)
    contrasts = np.asarray([objective(float(rate)) for rate in grid], dtype=np.float64)
    best_index = int(np.argmax(contrasts))
    step = float(grid[1] - grid[0])
    lower = max(-maximum_rate, float(grid[best_index] - step))
    upper = min(maximum_rate, float(grid[best_index] + step))
    optimized = minimize_scalar(
        lambda rate: -objective(float(rate)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 48},
    )
    rate = float(optimized.x)
    contrast, survival = _contrast_for_rate(
        rate,
        xy=points,
        times_s=times,
        polarities=polarity,
        image_shape=image_shape,
        center_xy=center,
        event_centers_xy=centers,
        minimum_survival_fraction=minimum_survival_fraction,
    )
    null_contrast = objective(0.0)
    relative_gain = (contrast - null_contrast) / max(abs(null_contrast), 1e-12)
    positive = rate >= minimum_positive_rate
    informative = relative_gain >= minimum_relative_contrast_gain
    valid = bool(positive and informative and survival >= minimum_survival_fraction)
    if not positive:
        reason = "non_approaching_best_warp"
    elif not informative:
        reason = "insufficient_contrast_gain"
    elif survival < minimum_survival_fraction:
        reason = "insufficient_warp_survival"
    else:
        reason = "ok"
    # Confidence is bounded and uses only objective sharpness/survival.
    second_best = float(np.partition(contrasts, -2)[-2])
    peak_margin = max(contrast - second_best, 0.0) / max(abs(contrast), 1e-12)
    confidence = float(np.clip(relative_gain, 0.0, 1.0) * survival * np.sqrt(peak_margin))
    return CMaxResult(
        inverse_ttc_per_s=rate if valid else float("nan"),
        ttc_seconds=(1.0 / rate) if valid else float("nan"),
        valid=valid,
        reason=reason,
        contrast=float(contrast),
        null_contrast=float(null_contrast),
        relative_contrast_gain=float(relative_gain),
        survival_fraction=float(survival),
        confidence=confidence if valid else 0.0,
        evaluations=evaluations,
    )


__all__ = [
    "CMaxResult",
    "image_of_warped_events",
    "iwe_contrast",
    "maximize_radial_event_contrast",
    "warp_radial_events",
]
