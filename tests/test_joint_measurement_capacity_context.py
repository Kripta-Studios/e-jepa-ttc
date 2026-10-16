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


def test_evaluation_adapter_restores_historical_paths(monkeypatch):
    from operational.joint_measurement import capacity_evaluate as adapter
    from operational.joint_measurement import export, geometry_cost, geometry_evaluate

    monkeypatch.setattr(adapter, "registered", lambda: {})
    original = (geometry_evaluate.DIRECTORY, geometry_cost.COST, export.RUN, export.initialize)
    with adapter.context():
        assert geometry_evaluate.DIRECTORY == capacity.DIRECTORY
        assert geometry_cost.COST == capacity.DIRECTORY / "cost/prepared_forward"
        assert export.RUN == capacity.DIRECTORY
        assert export.initialize is adapter.capacity_initialize
    assert original == (
        geometry_evaluate.DIRECTORY,
        geometry_cost.COST,
        export.RUN,
        export.initialize,
    )
