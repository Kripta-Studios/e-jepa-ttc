"""Prospective exploration thresholds and mandatory scientific safeguards."""

import copy

from operational.joint_measurement.evaluate import development_decision


def example():
    return {
        "overall_MiD": 100.0,
        "complete_prediction_and_mid_coverage": True,
        "diagnostics": {"sign_error_rate": 0.02},
        "bins": {
            name: {"group_macro_mid": 100.0} for name in ["crucial", "small", "large", "negative"]
        },
    }


def test_crucial_route_does_not_require_final_system_promotion():
    reference = example()
    candidate = copy.deepcopy(reference)
    candidate["overall_MiD"] = 101.5
    candidate["bins"]["crucial"]["group_macro_mid"] = 95.0
    decision = development_decision(reference, candidate)
    assert decision["eligible"]
    assert decision["status_scope"] == "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION"
    candidate["overall_MiD"] = 102.1
    assert not development_decision(reference, candidate)["eligible"]


def test_aggregate_gain_cannot_override_band_sign_or_coverage_failure():
    reference = example()
    candidate = copy.deepcopy(reference)
    candidate["overall_MiD"] = 90.0
    assert development_decision(reference, candidate)["eligible"]
    bad = copy.deepcopy(candidate)
    bad["bins"]["negative"]["group_macro_mid"] = 110.1
    assert not development_decision(reference, bad)["eligible"]
    bad = copy.deepcopy(candidate)
    bad["diagnostics"]["sign_error_rate"] = 0.0251
    assert not development_decision(reference, bad)["eligible"]
    bad = copy.deepcopy(candidate)
    bad["complete_prediction_and_mid_coverage"] = False
    assert not development_decision(reference, bad)["eligible"]
    bad["overall_MiD"] = None
    assert not development_decision(reference, bad)["eligible"]
