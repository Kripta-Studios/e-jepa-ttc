"""One fixed causal inverse-TTC transport control, with explicit resets."""

from __future__ import annotations

import numpy as np


def causal_transport_filter(
    predictions: np.ndarray,
    timestamps_us: np.ndarray,
    *,
    observation_weight: float = 0.5,
    maximum_gap_s: float = 0.5,
) -> np.ndarray:
    """Combine current inverse TTC with constant-speed propagation of prior output.

    Missing current predictions stay missing. Sign changes, gaps, and crossing contact
    reset to the current measurement. This does not detect arbitrary acceleration.
    """
    if predictions.shape != timestamps_us.shape or predictions.ndim != 1:
        raise ValueError("Prediction and clock vectors differ")
    if np.any(np.diff(timestamps_us) <= 0) or not 0 < observation_weight <= 1:
        raise ValueError("Clocks must increase and observation weight must be in (0,1]")
    result = predictions.astype(np.float64).copy()
    for i in range(1, len(result)):
        previous, current = result[i - 1], predictions[i]
        dt = (int(timestamps_us[i]) - int(timestamps_us[i - 1])) / 1e6
        propagated = previous - dt
        if (
            not np.isfinite(previous)
            or not np.isfinite(current)
            or current == 0
            or dt > maximum_gap_s
            or propagated == 0
            or np.sign(previous) != np.sign(propagated)
            or np.sign(current) != np.sign(propagated)
        ):
            continue
        q = observation_weight / current + (1 - observation_weight) / propagated
        result[i] = np.copysign(min(abs(1 / q), 60), q)
    return result
