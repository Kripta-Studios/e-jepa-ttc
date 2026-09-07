"""Deterministic time-rank thinning, independent of all target values."""

from __future__ import annotations

import numpy as np


def select_time_density(
    sequences: np.ndarray, anchor_us: np.ndarray, tokens: np.ndarray, *, per_sequence: int
) -> np.ndarray:
    """Keep endpoints and evenly spaced chronological ranks per sequence.

    Stable token tie-breaks make selection invariant to source row permutations.
    This thins queries, not the historical sensor dependencies of retained queries.
    """
    if (
        per_sequence < 2
        or sequences.ndim != 1
        or not (sequences.shape == anchor_us.shape == tokens.shape)
    ):
        raise ValueError("aligned input-only vectors and cap >= 2 required")
    if anchor_us.dtype.kind not in "iu" or len(np.unique(tokens)) != len(tokens):
        raise ValueError("integer timestamps and unique query identities required")
    selected = []
    for sequence in np.unique(sequences):
        rows = np.flatnonzero(sequences == sequence)
        rows = rows[np.lexsort((tokens[rows], anchor_us[rows]))]
        count = min(per_sequence, len(rows))
        ranks = (
            np.arange(count, dtype=np.int64) * (len(rows) - 1) // (count - 1)
            if count > 1
            else np.zeros(count, dtype=np.int64)
        )
        selected.extend(rows[ranks].tolist())
    return np.asarray(selected, dtype=np.int64)
