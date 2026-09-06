"""Identity-aligned T2 factor effects for full or technically D0-only campaigns."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .evaluation import factorial_contrasts


def paired_factor_effects(
    predictions: pd.DataFrame, expected_queries: pd.DataFrame, *, d1_available: bool
) -> pd.DataFrame:
    """Return per-query effects for later sequence/track-aware uncertainty analysis.

    expected_queries must come from acknowledged OLD_DEV role metadata, with one
    sample_token, sequence_id, track_id and outer_fold per canonical query. This
    function cannot establish that authority itself. No incomplete-case dropping
    or positional pairing is permitted. Missing D1 yields H/C/HxC only, not an
    invented D estimate. Controls are not factorial cells.
    """
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold"]
    if (
        type(d1_available) is not bool
        or not set(identity) <= set(expected_queries)
        or len(expected_queries) != 8192
        or expected_queries.sample_token.duplicated().any()
        or expected_queries.loc[:, identity].isna().to_numpy().any()
        or set(expected_queries.outer_fold) != {0, 1, 2}
    ):
        raise ValueError("complete authoritative 8192-query cohort required")
    if not {*identity, "arm", "seed", "loss"} <= set(predictions):
        raise ValueError("missing factorial prediction fields")
    reference = expected_queries.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
    losses = {}
    for d, h, c in product((0, 1) if d1_available else (0,), (0, 1), (0, 1)):
        arm = f"TPR-D{d}-H{(1, 8)[h]}-C{(64, 160)[c]}"
        cell = predictions.loc[(predictions.arm == arm) & (predictions.seed == 7)]
        if len(cell) != 8192 or cell.sample_token.duplicated().any():
            raise ValueError(f"incomplete or duplicate factorial cell: {arm}")
        cell = cell.sort_values("sample_token").reset_index(drop=True)
        if not cell[identity].equals(reference):
            raise ValueError(f"query/sequence/track/fold pairing changed: {arm}")
        values = cell.loss.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("finite nonnegative per-query benchmark loss required")
        losses[d, h, c] = values
    if d1_available:
        effects = factorial_contrasts(losses)
    else:
        effects = {
            "H": ((losses[0, 1, 0] - losses[0, 0, 0]) + (losses[0, 1, 1] - losses[0, 0, 1])) / 2,
            "C": ((losses[0, 0, 1] - losses[0, 0, 0]) + (losses[0, 1, 1] - losses[0, 1, 0])) / 2,
            "HxC": losses[0, 1, 1] - losses[0, 1, 0] - losses[0, 0, 1] + losses[0, 0, 0],
        }
    return reference.assign(**effects)
