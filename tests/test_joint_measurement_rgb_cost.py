"""Cost signals cannot bypass coverage or scientific provenance checks."""

import copy

import pytest

from operational.joint_measurement import rgb_cost
from operational.joint_measurement.common import config
from operational.range_transfer.common import digest


def metric():
    return {
        "eligible": False,
        "checks": {"coverage": True, "sign": True, "bands": True},
        "overall_ratio": 1.01,
    }


def test_cost_signal_requires_registered_gain_and_overall_limit():
    routes = config()["selection"]["routes"]
    assert rgb_cost.combine_decision(metric(), 0.69, routes)["eligible"]
    assert not rgb_cost.combine_decision(metric(), 0.71, routes)["eligible"]
    outcome = metric()
    outcome["overall_ratio"] = 1.021
    assert not rgb_cost.combine_decision(outcome, 0.5, routes)["eligible"]


@pytest.mark.parametrize("guard", ["coverage", "sign", "bands", "missing"])
def test_speed_cannot_override_scientific_guard(guard):
    outcome = metric()
    if guard == "missing":
        del outcome["checks"]
    else:
        outcome["checks"][guard] = False
    assert not rgb_cost.combine_decision(outcome, 0.01, config()["selection"]["routes"])["eligible"]


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_cost_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        rgb_cost.combine_decision(metric(), ratio, config()["selection"]["routes"])


@pytest.mark.parametrize("change", ["binding", "query", "route", "nonfinite", "zero_time"])
def test_foreign_or_invalid_timing_fragment_rejected(change):
    binding = {"indices": [17], "endpoints": {"RGB": "weight-hash"}}
    row = {
        "binding": digest(binding),
        "P_index": 17,
        "timings": {
            "RGB": {"seconds": 0.01, "ttc_seconds": 4.0, "peak_extra_allocated_bytes": 100}
        },
    }
    rgb_cost.validate_row(row, binding, 0)
    bad = copy.deepcopy(row)
    if change == "binding":
        bad["binding"] = "another-binding"
    elif change == "query":
        bad["P_index"] = 18
    elif change == "route":
        bad["timings"]["another-model"] = bad["timings"].pop("RGB")
    elif change == "nonfinite":
        bad["timings"]["RGB"]["ttc_seconds"] = float("nan")
    else:
        bad["timings"]["RGB"]["seconds"] = 0
    with pytest.raises(ValueError, match="fragment"):
        rgb_cost.validate_row(bad, binding, 0)
