"""Capacity execution must not leak its namespace into the historical geometry engine."""

import pytest

from operational.joint_measurement import capacity_train as capacity
from operational.joint_measurement import geometry_train as geometry


def test_capacity_context_restores_historical_engine_after_exception():
    fields = (
        "FIT",
        "DIRECTORY",
        "REGISTRATION",
        "registered",
        "GeometricPairMeasurement",
        "backward",
    )
    before = {name: getattr(geometry, name) for name in fields}
    with pytest.raises(RuntimeError, match="test interruption"):
        with capacity.engine_context(capture=True):
            assert geometry.DIRECTORY == capacity.DIRECTORY
            assert geometry.FIT == capacity.FIT
            assert geometry.backward is capacity.captured_backward
            with capacity.engine_context(capture=False):
                assert geometry.backward is capacity.backward
            assert geometry.backward is capacity.captured_backward
            raise RuntimeError("test interruption")
    assert before == {name: getattr(geometry, name) for name in fields}
