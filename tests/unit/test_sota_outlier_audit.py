"""Hand-calculated post-hoc sensitivity preserves targets and complete cohorts."""

from __future__ import annotations

import numpy as np
import pytest

from operational.sota_eval.outlier_audit import metrics, output_bound, sensitivity


def test_sensitivity_uses_same_bound_without_filtering_or_clipping_targets():
    truth = np.array([2.0, -4.0, 100.0])
    sequence = np.array(["one", "one", "two"])
    prediction = np.array([200.0, -100.0, 120.0])
    original = prediction.copy()
    rows = sensitivity(truth, {"A": prediction, "B": prediction.copy()}, sequence, 60.0)
    assert len(rows) == 4
    for row in rows:
        assert row["rows"] == 3
        assert row["targets_outside_bound_retained"] == 1
        assert row["predictions_outside_bound"] == 3
    bounded = rows[1]
    assert bounded["analysis"] == "POSTHOC_COMMON_OUTPUT_BOUND"
    assert bounded["micro_mae_seconds"] == pytest.approx((58 + 56 + 40) / 3)
    assert bounded["micro_rte_percent"] == pytest.approx((2900 + 1400 + 40) / 3)
    assert bounded["macro_sequence_mae_seconds"] == pytest.approx((57 + 40) / 2)
    assert rows[1]["micro_mae_seconds"] == rows[3]["micro_mae_seconds"]
    np.testing.assert_array_equal(prediction, original)
    np.testing.assert_array_equal(truth, [2, -4, 100])


def test_nonfinite_predictions_fail_instead_of_hiding_rows():
    with pytest.raises(ValueError, match="Complete finite"):
        metrics(np.array([1.0, 2.0]), np.array([1.0, np.nan]), np.array(["s", "s"]))
    with pytest.raises(ValueError, match="Complete finite"):
        metrics(np.array([0.0]), np.array([1.0]), np.array(["s"]))


def test_output_support_is_derived_from_source_not_a_tuned_threshold(tmp_path):
    path = tmp_path / "phase.py"
    path.write_text(
        "def phase_to_ttc(phase):\n    return sign / q.abs().clamp_min(1.0 / 60.0)\n",
        encoding="utf-8",
    )
    assert output_bound(path) == 60
    path.write_text(
        "def phase_to_ttc(phase):\n    return sign / q.abs().clamp_min(0.002)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="re-audited|reciprocal"):
        output_bound(path)
