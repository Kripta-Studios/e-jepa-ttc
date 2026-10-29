"""The interrupted pilot's serial tail cannot spend beyond its original reservation."""

import json
from pathlib import Path

import pytest

from operational.train40_system import engine_parallel_tail_v4 as tail


@pytest.mark.parametrize(
    "committed,allowed", [(920, True), (1419, True), (1420, False), (1421, False)]
)
def test_tail_stops_at_original_target(tmp_path: Path, monkeypatch, committed, allowed):
    (tmp_path / "fits/c2f_seed7").mkdir(parents=True)
    (tmp_path / "PARALLEL_PILOT_V4_LEDGER.json").write_text(
        json.dumps({"arms": {"c2f": {"target_committed": 1420}}})
    )
    (tmp_path / "fits/c2f_seed7/UPDATE_JOURNAL.json").write_text(
        json.dumps({"committed_updates": committed})
    )
    monkeypatch.setattr(
        tail.resource_monitor.ResourceMonitor, "guard", lambda self, output: (True, {"reasons": []})
    )
    monitor = object.__new__(tail.TailMonitor)
    result, telemetry = monitor.guard(tmp_path)
    assert result is allowed
    assert ("PARALLEL_ARM_TARGET_REACHED" in telemetry["reasons"]) is (not allowed)


@pytest.mark.parametrize("committed", [1419, 1420])
def test_tail_keeps_existing_resource_and_pause_denial(tmp_path: Path, monkeypatch, committed):
    (tmp_path / "fits/c2f_seed7").mkdir(parents=True)
    (tmp_path / "PARALLEL_PILOT_V4_LEDGER.json").write_text(
        json.dumps({"arms": {"c2f": {"target_committed": 1420}}})
    )
    (tmp_path / "fits/c2f_seed7/UPDATE_JOURNAL.json").write_text(
        json.dumps({"committed_updates": committed})
    )
    monkeypatch.setattr(
        tail.resource_monitor.ResourceMonitor,
        "guard",
        lambda self, output: (False, {"reasons": ["STOP_REQUESTED", "RAM_RESOURCE_BOUNDARY"]}),
    )
    monitor = object.__new__(tail.TailMonitor)
    result, telemetry = monitor.guard(tmp_path)
    assert result is False
    assert "STOP_REQUESTED" in telemetry["reasons"]
    assert "RAM_RESOURCE_BOUNDARY" in telemetry["reasons"]
