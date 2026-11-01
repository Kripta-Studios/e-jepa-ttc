"""Persistence must share the same causal admission and frame validation as the clock."""

import pytest

from operational.release_research.physics import Measurement
from scripts.release_clock_cadence import clock_readout


def test_persistence_keeps_latest_causal_measurement_without_transport():
    history = [
        Measurement(1.0, 1.0, 0.3, 0.01, "a"),
        Measurement(1.1, 1.1, 0.4, 0.01, "a"),
        Measurement(1.15, 1.25, 5.0, 0.01, "a"),
    ]
    value = clock_readout(history, 1.2, "a", 0.8, "persistence")
    assert value.q == 0.4
    assert value.observations == 2
    assert clock_readout(history, 1.2, "a", 0.8, "transport").q > 0.4


def test_persistence_does_not_recover_stale_or_invalid_frames():
    old = [Measurement(0.0, 0.0, 0.3, 0.01, "a")]
    value = clock_readout(old, 1.0, "a", 0.8, "persistence")
    assert value.q == 0.8
    assert value.reason == "no_causal_history"
    with pytest.raises(ValueError, match="ROI"):
        clock_readout([Measurement(0.9, 0.9, 0.3, 0.01, "b")], 1.0, "a", 0.8, "persistence")


def test_persistence_rejects_duplicate_evidence_like_the_observer():
    measurement = Measurement(0.9, 0.9, 0.3, 0.01, "a")
    with pytest.raises(ValueError, match="Duplicate"):
        clock_readout([measurement, measurement], 1.0, "a", 0.8, "persistence")
