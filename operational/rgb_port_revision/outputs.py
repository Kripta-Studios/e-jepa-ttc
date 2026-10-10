"""Correct signed phase interval transforms."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc


def interval_from_phase(
    low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the central phase interval on one branch; crossing zero is unavailable.

    These are interval bounds, not unconditional TTC quantiles: a signed phase
    distribution need not have all its mass on the same monotone branch.
    """
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != high.shape or np.any(low > high):
        raise ValueError("Phase interval endpoints must be aligned and ordered")
    finite = np.isfinite(low) & np.isfinite(high)
    same = finite & (((low > 0) & (high > 0)) | ((low < 0) & (high < 0)))
    lower = phase_to_ttc(torch.as_tensor(high).float()).numpy().astype(np.float64)
    upper = phase_to_ttc(torch.as_tensor(low).float()).numpy().astype(np.float64)
    lower[~same], upper[~same] = np.nan, np.nan
    status = np.where(same, "SAME_BRANCH_INTERVAL", "CROSSES_ZERO_UNAVAILABLE")
    status = np.where(finite, status, "NONFINITE_PHASE_INTERVAL")
    return lower, upper, status
