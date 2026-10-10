"""Regression coverage for missing predictions and diagnostic signed errors."""

import numpy as np
import pytest

from operational.garl_comparison.report import measure
from operational.garl_comparison.test_event import check_row


def test_missing_prediction_does_not_improve_full_cohort() -> None:
    value = measure(np.array([1.0, 2.0]), np.array([1.0, np.nan]))
    assert value["coverage"] == 0.5
    assert value["mae_s"] is None


def test_signed_bias_is_not_absolute_error() -> None:
    value = measure(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert value["bias_s"] == 0
    assert value["mae_s"] == 1
    assert value["underestimate_percent"] == 50


def test_invalid_infinity_is_preserved_as_failure() -> None:
    value = measure(np.array([1.0, 2.0]), np.array([np.inf, 2.0]))
    assert value["finite_n"] == 1
    assert value["p95_ae_s"] is None


def test_negative_prediction_is_not_an_urgent_warning() -> None:
    value = measure(np.array([0.5, 0.8, 1.0]), np.array([-1.0, np.nan, 0.5]))
    assert value["urgent_gt_n"] == 3
    assert value["urgent_miss_n"] == 2
    assert value["urgent_miss_percent"] == pytest.approx(200 / 3)


def test_zero_truth_requires_explicit_eligibility() -> None:
    with pytest.raises(ValueError, match="zero"):
        measure(np.array([0.0]), np.array([1.0]))


def test_native_test_input_rejects_training_or_unmatched_path() -> None:
    row = {
        "sequence_id": "example",
        "events_path": "data/train/example/events.h5",
        "event_windows_us": [[0, 100000], [100000, 200000]],
        "boxes_xyxy": [[0, 0, 10, 10], [0, 0, 12, 12]],
    }
    with pytest.raises(ValueError, match="official test"):
        check_row(row)
    row["events_path"] = "data/test/example/events.h5"
    check_row(row)
    row["event_windows_us"] = [[0, 100000]]
    with pytest.raises(ValueError, match="two native event"):
        check_row(row)
