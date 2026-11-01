"""Submission training exclusions and organizer-confirmed evaluation processing."""

from __future__ import annotations

import numpy as np


def interpolate_negative_predictions(
    times: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Interpolate interior negatives for evaluation only; never change raw results."""
    times = np.asarray(times, dtype=np.float64)
    original = np.asarray(predictions, dtype=np.float64)
    if times.ndim != 1 or original.shape != times.shape or not np.isfinite(times).all():
        raise ValueError("Expected matching finite one-dimensional timestamps/predictions")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    if not np.isfinite(original).all():
        raise ValueError("Cannot infer organizer handling of missing predictions")
    bad = original < 0
    changed = np.flatnonzero(bad).tolist()
    result = original.copy()
    if bad.any():
        if bad[0] or bad[-1]:
            raise ValueError("Negative endpoint has no surrounding valid predictions")
        result[bad] = np.interp(times[bad], times[~bad], original[~bad])
    return result, changed


def eligible_population(a: dict[str, np.ndarray], excluded: set[str]) -> dict[str, np.ndarray]:
    """Filter benchmark Slider labels and reject any matched road recording."""
    if set(a["group"].tolist()) & excluded:
        raise ValueError("Input population contains benchmark road overlap")
    mask = ~np.isin(a["group"], ["Slider-750", "Slider-1000"])
    keys = ("native", "target", "group", "family", "token", "geometry", "quality")
    return {k: a[k][mask].copy() for k in keys}
