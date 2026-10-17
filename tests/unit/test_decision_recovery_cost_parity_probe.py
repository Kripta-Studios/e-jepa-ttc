"""CPU-only arithmetic checks for the directed cost parity probe."""

import math

import pytest

from operational.decision_recovery.cost_parity_probe import (
    ARMS,
    INDEX,
    NEIGHBORS,
    POSITION,
    REPEATS,
    compare,
    validate_registration_identity,
)


def test_compare_retains_direction_saturation_and_exact_differences():
    result = compare(-60.0, 30.0)
    assert result["absolute_difference"] == 90.0
    assert result["relative_to_right"] == 3.0
    assert result["same_sign"] is False
    assert result["left_saturated"] is True
    assert result["right_saturated"] is False


def test_compare_exact_is_finite_zero():
    result = compare(0.025, 0.025)
    assert result["absolute_difference"] == 0
    assert result["relative_to_right"] == 0
    assert math.isfinite(result["relative_to_right"])


def test_registration_identity_is_fixed_and_zero_update():
    value = {
        "schema": "v19_cost_batch_parity_probe_v1",
        "index": INDEX,
        "neighbors": list(NEIGHBORS),
        "position": POSITION,
        "repeats": REPEATS,
        "optimizer_updates": 0,
        "backward": False,
        "endpoints": {arm: {} for arm in ARMS},
    }
    validate_registration_identity(value)
    value["index"] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_registration_identity(value)
