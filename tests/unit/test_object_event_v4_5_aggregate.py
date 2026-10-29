from __future__ import annotations

import pandas as pd
import pytest

from scripts.aggregate_object_event_v4_5_multiseed import _align, _pairwise


def _frame(seed: int, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["a", "a", "b", "b"],
            "sample_token": ["0", "1", "0", "1"],
            "track_id": ["x", "x", "y", "y"],
            "delta_t_s": [0.1] * 4,
            "target_ttc_s": [2.0, 5.0, -4.0, 8.0],
            "target_expansion": [0.05, 0.02, -0.025, 0.0125],
            "prediction_expansion": [0.04, 0.01, -0.02, 0.01 + offset],
            "zero_events_expansion": [0.0] * 4,
            "shuffled_mean_expansion": [0.0] * 4,
            "seed": [seed] * 4,
        }
    )


def test_alignment_builds_mean_and_seed_dispersion() -> None:
    aligned = _align({7: _frame(7), 13: _frame(13, offset=0.02), 23: _frame(23)})
    assert len(aligned) == 4
    assert aligned.loc[3, "ensemble_expansion"] == pytest.approx((0.01 + 0.03 + 0.01) / 3)
    assert aligned.loc[3, "seed_prediction_std"] > 0.0


def test_alignment_fails_closed_when_targets_change() -> None:
    bad = _frame(13)
    bad.loc[0, "target_expansion"] = 0.20
    with pytest.raises(ValueError, match="target_expansion"):
        _align({7: _frame(7), 13: bad, 23: _frame(23)})


def test_pairwise_reports_high_agreement_for_close_seeds() -> None:
    aligned = _align({7: _frame(7), 13: _frame(13, offset=0.001), 23: _frame(23)})
    pairwise = _pairwise(aligned, (7, 13, 23))
    assert len(pairwise) == 3
    assert pairwise["prediction_pearson"].min() > 0.99
    assert pairwise["sign_agreement"].min() == 1.0
