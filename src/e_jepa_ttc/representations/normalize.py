"""Numerically explicit normalization for event representations."""

from __future__ import annotations

import numpy as np


def normalize_window(
    values: np.ndarray,
    *,
    robust_percentile: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Center and scale a representation using a per-window robust magnitude."""

    if not 0.0 < robust_percentile <= 100.0 or eps <= 0.0:
        raise ValueError("robust_percentile must be in (0,100] and eps positive.")
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Representation contains non-finite values.")
    scale = float(np.percentile(np.abs(array), robust_percentile))
    if scale < eps:
        return np.zeros_like(array)
    return (array / scale).astype(np.float32, copy=False)


__all__ = ["normalize_window"]
