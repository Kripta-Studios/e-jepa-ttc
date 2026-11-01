"""Prediction sensitivity never substitutes missing values or asserts historical accuracy."""

import pytest

from scripts.continue_native_platform_diagnostic import validate_profile
from scripts.release_native_platform_pair import sensitivity


def test_sensitivity_keeps_missing_outputs_and_signed_changes():
    result = sensitivity([-2.0, 1.0, None, 4.0], [-2.0, -1.0, 3.0, None])
    assert result["paired"] == 2
    assert result["exact_predictions"] == 1
    assert result["newly_missing"] == 1
    assert result["newly_available"] == 1
    assert result["sign_changes"] == 1
    assert result["maximum_absolute_change_seconds"] == 2
    assert result["promotion_admitted"] is False
    assert result["historical_parity_granted"] is False
    assert result["native_accuracy_evaluated"] is False


def test_no_paired_outputs_does_not_invent_zero_change():
    result = sensitivity([None], [None])
    assert result["paired"] == 0
    assert result["maximum_absolute_change_seconds"] is None


def test_misaligned_predictions_cannot_be_compared():
    with pytest.raises(ValueError):
        sensitivity([1.0], [1.0, 2.0])


@pytest.mark.parametrize(
    "parity",
    [
        {},
        {"one": {"fields": {}}},
        {"one": {"fields": {}}, "two": {"fields": {}}},
        {
            "one": {"fields": {"head.output_ttc": {"passed": False}}},
            "two": {"fields": {"head.output_ttc": {"passed": True}}},
        },
    ],
)
def test_incomplete_or_failed_ttc_profile_cannot_admit_local_diagnostic(parity):
    with pytest.raises(ValueError):
        validate_profile(parity)


def test_local_diagnostic_retains_internal_failure_without_reclassifying_it():
    parity = {
        name: {
            "fields": {"head.output_ttc": {"passed": True}, "context.input_0": {"passed": False}}
        }
        for name in ("one", "two")
    }
    validate_profile(parity)
    assert all(not row["fields"]["context.input_0"]["passed"] for row in parity.values())
