"""Diagnostic subsets retain full-cohort mass and do not replace the primary score."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t.diagnostic_summary import summarize_diagnostics


def fixture_frame():
    index = np.arange(8192)
    return pd.DataFrame(
        dict(
            sample_token=index.astype(str),
            sequence_id=(index % 9).astype(str),
            outer_fold=index % 3,
            target_ttc=np.asarray([-2.0, 1.0, 4.0, 8.0])[index % 4],
            arm="TPR-D0-H8-C160",
            seed=7,
            history_count=np.where(index % 2 == 0, 1, 8),
            history_span_us=np.where(index % 2 == 0, 0, 350000),
            roi_age_us=10000,
            cold_start=index % 2 == 0,
            hull_position=np.where(index % 2 == 0, "inside", "above"),
            loss=10.0,
            signed_phase_error=0.001,
            escape_gain=0.0,
            escape_harm=0.0,
            phase_interval_width=0.1,
            wrong_sign=False,
            phase_interval_covers=True,
            phase_support_saturation=False,
            target_outside_support=False,
            finite_ttc_cap=False,
        )
    )


def test_diagnostic_mass_and_empty_strata():
    report = summarize_diagnostics(fixture_frame(), history_length=8)
    primary = report.loc[report.axis == "all"].iloc[0]
    assert primary.queries == 8192
    assert primary.loss_global_contribution == pytest.approx(10)
    for axis in ("sequence_id", "history_count", "history_span_us", "hull_position", "ttc_bucket"):
        subset = report.loc[report.axis == axis]
        assert subset.queries.sum() == 8192
        assert subset.loss_global_contribution.sum() == pytest.approx(10)
    empty = report.loc[report.stratum == "partial_history"].iloc[0]
    assert empty.queries == 0 and empty.loss_global_contribution == 0
    assert pd.isna(empty.loss_conditional_weighted_mean)


@pytest.mark.parametrize("mode", ["missing", "count", "cold", "loss", "seed"])
def test_reject_inconsistent_diagnostics(mode):
    frame = fixture_frame()
    if mode == "missing":
        frame = frame.iloc[:-1]
    elif mode == "count":
        frame.loc[0, "history_count"] = 16
    elif mode == "cold":
        frame.loc[0, "cold_start"] = False
    elif mode == "loss":
        frame.loc[0, "loss"] = np.nan
    else:
        frame.loc[0, "seed"] = 23
    with pytest.raises(ValueError):
        summarize_diagnostics(frame, history_length=8)
