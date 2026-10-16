"""A complete cost claim must survive recomputed domain, parity and population checks."""

import copy

import pytest

from operational.joint_measurement.closure_evidence import ROUTES, validate_cost_rows
from operational.range_transfer.common import digest


def fixture_rows():
    registration = {"indices": list(range(64)), "parity_rtol": 0, "parity_atol": 0.001}
    timing = {
        "prediction": 1.0,
        "reference": 1.0,
        "parity": True,
        "same_sign": True,
        "valid_native_outputs": True,
        "strict_parity": True,
        "seconds": 0.1,
        "absolute_difference": 0.0,
    }
    rows = [
        {
            "index": index,
            "binding": digest(registration),
            "timings": {name: copy.deepcopy(timing) for name in ROUTES},
        }
        for index in range(64)
    ]
    return registration, rows


def test_closure_preserves_strict_failure_under_amended_acceptance():
    registration, rows = fixture_rows()
    timing = rows[0]["timings"]["F_TRUE"]
    timing.update(prediction=1.0008, absolute_difference=1.0008 - 1.0, strict_parity=False)
    result = validate_cost_rows(registration, rows)
    assert result["strict_failures"] == 1
    assert result["amended_acceptance_pass"]
    assert result["paired_outputs"] == 256


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "route", "sign", "nan", "tolerance"])
def test_closure_rejects_incomplete_or_falsely_accepted_cost(mutation):
    registration, rows = fixture_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "route":
        rows[0]["timings"].pop("F_TRUE")
    elif mutation == "sign":
        rows[0]["timings"]["F_TRUE"]["prediction"] = -1.0
    elif mutation == "nan":
        rows[0]["timings"]["F_TRUE"]["prediction"] = float("nan")
    else:
        registration["parity_atol"] = 0.002
    with pytest.raises(ValueError):
        validate_cost_rows(registration, rows)
