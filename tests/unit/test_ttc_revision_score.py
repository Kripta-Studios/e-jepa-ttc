"""Prevent favourable filtering, sign mistakes and window bootstrap regressions."""

import numpy as np

from operational.ttc_revision.score import metrics, paired_interval


def test_missing_or_infinite_prediction_invalidates_complete_cohort_mean():
    for missing in (np.nan, np.inf, -np.inf):
        result = metrics(np.array([1.0, 2.0]), np.array([1.0, missing]))
        assert result["coverage"] == 0.5
        assert result["status"] == "INCOMPLETE"
        assert "mae" not in result


def test_urgent_negative_output_counts_as_missed_warning():
    result = metrics(np.array([0.5, 0.8, 2.0]), np.array([-0.5, 1.2, 2.0]))
    assert result["urgent_miss_fraction"] == 1
    assert np.isclose(result["mae"], 1.4 / 3)
    assert result["sign_error_fraction"] == 1 / 3
    alarms = metrics(np.array([-2.0, 0.5, 2.0, 4.0]), np.array([0.5, 0.6, 0.9, -1.0]))
    assert alarms["urgent_miss_fraction"] == 0
    assert alarms["urgent_false_alarm_fraction"] == 2 / 3


def test_paired_bootstrap_resamples_groups_not_windows():
    truth = np.ones(100)
    left = np.r_[np.ones(99), 101.0]
    groups = np.array(["large"] * 99 + ["small"])
    result = paired_interval(truth, left, truth, groups, draws=10000)
    assert result["difference_mae"] == 1
    assert result["ci_low"] == 0
    assert result["ci_high"] == 100
    assert result["groups"] == 2
