"""CPU-only checks for the directed V4 failure probe."""

from operational.decision_recovery.cost_v4_failure_probe import (
    BATCH_INDICES,
    INDEX,
    ORDINAL,
    POSITION,
    comparison,
)


def test_fixed_failure_location_and_float_witness():
    assert (ORDINAL, INDEX, BATCH_INDICES, POSITION) == (42, 91, (88, 89, 90, 91), 3)
    exact = comparison(0.25, 0.25)
    assert exact["exact"] is True
    assert exact["absolute_difference"] == 0
    assert exact["left_float32_bits"] == exact["right_float32_bits"] == "3e800000"
    drift = comparison(0.25, 0.25000003)
    assert drift["exact"] is False and drift["absolute_difference"] > 0
