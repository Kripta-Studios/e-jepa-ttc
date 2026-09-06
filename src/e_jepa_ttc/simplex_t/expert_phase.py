"""Companion-only extended TTC coordinate; historical source rules stay intact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc
from e_jepa_ttc.models.three_expert_router import BASE8_FEATURES, PHASE17_FEATURES


def expert_phase_from_ttc(values: np.ndarray) -> np.ndarray:
    """Preserve finite historical phases; map signed infinity to its zero limit.

    This is for original expert predictions, never supervision. NaN, zero and
    finite TTC in (0,.1] remain errors. No TTC point is clipped or replaced.
    """
    values = np.asarray(values, dtype=np.float64)
    infinite = np.isinf(values)
    result = np.empty_like(values)
    result[~infinite] = phase_from_ttc(values[~infinite])
    result[infinite] = np.copysign(0.0, values[infinite])
    return result


def expert_benchmark_phase(values: np.ndarray) -> np.ndarray:
    """Retain the original NumPy scoring arithmetic, extending only infinity."""
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).any() or not ((values < 0) | (values > 0.1)).all():
        raise ValueError("expert TTC outside the extended benchmark-phase domain")
    return -np.log1p(-0.1 / values)


def build_expert_features(
    a5: pd.DataFrame, c2f: pd.DataFrame, pair_prediction_ttc: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact historical PHASE17 operations with an explicit infinite-TTC limit."""
    if a5.token_id.astype(str).tolist() != c2f.token_id.astype(str).tolist():
        raise ValueError("A5/C2F tokens are not row-aligned")
    base = pd.DataFrame(
        {
            name: np.asarray(
                a5[name] if name.startswith(("shared_", "a5_")) else c2f[name], dtype=np.float64
            )
            for name in BASE8_FEATURES
        }
    )
    a = expert_phase_from_ttc(a5.prediction_ttc.to_numpy(np.float64))
    c = expert_phase_from_ttc(c2f.prediction_ttc.to_numpy(np.float64))
    p = expert_phase_from_ttc(np.asarray(pair_prediction_ttc, np.float64))
    phase = pd.DataFrame(
        {
            "a5_benchmark_phase": a,
            "c2f_benchmark_phase": c,
            "pair_benchmark_phase": p,
            "pair_minus_a5_phase": p - a,
            "pair_minus_c2f_phase": p - c,
            "c2f_minus_a5_phase": c - a,
            "abs_pair_minus_a5_phase": np.abs(p - a),
            "abs_pair_minus_c2f_phase": np.abs(p - c),
            "abs_c2f_minus_a5_phase": np.abs(c - a),
        }
    )
    combined = pd.concat((base, phase), axis=1)
    if tuple(combined.columns) != PHASE17_FEATURES:
        raise AssertionError("companion feature order drifted")
    return base, combined
