"""Sampled adjacency is input-only and does not imply tracked object history."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t.temporal_diagnostics import sampled_temporal_diagnostics


def fixture():
    i = np.arange(8192)
    return pd.DataFrame(
        {
            "sample_token": [f"q{n}" for n in i],
            "sequence_id": (i % 9).astype(str),
            "track_id": "a",
            "outer_fold": i % 9 % 3,
            "anchor_us": i // 9 * 100_000,
            "arm": "TPR-D0-H8-C160",
            "seed": 7,
            "target_ttc": np.array([-2.0, 1.0, 4.0, 8.0])[i // 9 % 4],
            "prediction_ttc_s": 2.0,
            "loss": 1.0,
        }
    )


def test_adjacency_independent_of_targets_and_order():
    frame = fixture()
    rows, summary = sampled_temporal_diagnostics(frame)
    frame.target_ttc *= 0.99
    changed, _ = sampled_temporal_diagnostics(frame.sample(frac=1, random_state=7))
    fields = ["sample_token", "previous_sample_token", "sample_gap_us", "temporal_eligible"]
    pd.testing.assert_frame_equal(rows[fields], changed[fields])
    assert rows.temporal_eligible.sum() == 8183
    assert rows.sign_transition.sum() > 0
    # Sign transitions and phase-rate cutpoints are distinct diagnostics.
    assert rows.rapid_change.sum() == 2052
    assert rows.sign_transition.sum() == 4095
    assert summary.set_index("stratum").loc[
        "all_queries", "loss_global_contribution"
    ] == pytest.approx(1)


def test_gap_does_not_bridge_or_drop_queries():
    frame = fixture()
    frame.loc[frame.anchor_us >= 100_000, "anchor_us"] += 1_000_000
    rows, summary = sampled_temporal_diagnostics(frame)
    assert len(rows) == 8192
    assert (rows.temporal_ineligible_reason == "GAP_EXCEEDS_FIXED_SPAN").sum() == 9
    assert summary.set_index("stratum").loc["all_queries", "queries"] == 8192


def test_empty_subset_and_infinite_selector_sign():
    frame = fixture()
    frame.track_id = frame.sample_token
    frame.prediction_ttc_s = np.inf
    rows, summary = sampled_temporal_diagnostics(frame)
    assert np.array_equal(rows.current_sign_correct, rows.target_ttc > 0)
    empty = summary.set_index("stratum").loc["sign_transition"]
    assert empty["empty"] and empty.loss_global_contribution == 0
    assert pd.isna(empty.loss_conditional_weighted_mean)


@pytest.mark.parametrize("bad", ["duplicate_time", "float_time", "target_nan", "fold", "missing"])
def test_reject_ambiguous_inputs(bad):
    frame = fixture()
    if bad == "duplicate_time":
        frame.loc[9, "anchor_us"] = 0
    elif bad == "float_time":
        frame.anchor_us = frame.anchor_us.astype(float)
    elif bad == "target_nan":
        frame.loc[0, "target_ttc"] = np.nan
    elif bad == "fold":
        frame.loc[0, "outer_fold"] = 1
    else:
        frame = frame.drop(columns="anchor_us")
    with pytest.raises(ValueError):
        sampled_temporal_diagnostics(frame)
