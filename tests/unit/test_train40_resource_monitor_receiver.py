"""CPU-only tests for continuous resource-monitor pipe draining."""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from operational.train40_system import resource_monitor
from operational.train40_system.engine_c2f_graph_replay import C2FResourceMonitor
from operational.train40_system.resource_monitor_receiver import (
    ContinuousC2FResourceMonitor,
)


class FakeConnection:
    def __init__(self) -> None:
        self.values: deque[object] = deque()
        self.lock = Lock()
        self.closed = False
        self.error: BaseException | None = None

    def append(self, value: object) -> None:
        with self.lock:
            self.values.append(value)

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        with self.lock:
            if self.error is not None:
                raise self.error
            return bool(self.values)

    def recv(self) -> object:
        with self.lock:
            return self.values.popleft()

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    pid = 321

    def __init__(self) -> None:
        self.alive = True
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        del timeout
        self.join_calls += 1
        self.alive = False

    def terminate(self) -> None:
        self.alive = False


class FakeStop:
    def __init__(self) -> None:
        self.set_calls = 0

    def set(self) -> None:
        self.set_calls += 1


def _sample(sequence: int, sampled: float) -> dict[str, Any]:
    return {
        "status": "OK",
        "sequence": sequence,
        "sampled_monotonic": sampled,
        "root_pid": 777,
        "root_create_time": 12.5,
        "tree_rss_bytes": 1_000,
        "host_available_bytes": 9_000_000_000,
        "disk_free_bytes": 30_000_000_000,
        "other_heavy_processes": [],
        "inventory_unreadable_python_pids": [],
        "torch_imported": False,
    }


def _output(tmp_path: Path) -> Path:
    (tmp_path / "fits/c2f_seed7").mkdir(parents=True)
    (tmp_path / "AUTHORIZATION.json").write_text(
        json.dumps(
            {
                "previous_physical_execution_upper": 0,
                "deadline_utc": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "TECHNICAL_ACCOUNTING.json").write_text(
        json.dumps({"synthetic_optimizer_updates": 0}), encoding="utf-8"
    )
    (tmp_path / "fits/c2f_seed7/UPDATE_JOURNAL.json").write_text(
        json.dumps(
            {
                "committed_updates": 0,
                "recovery_upper": 0,
                "pending_update_upper": 0,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _armed(output: Path) -> tuple[ContinuousC2FResourceMonitor, FakeConnection]:
    monitor = ContinuousC2FResourceMonitor(output)
    connection = FakeConnection()
    monitor._receiver = connection
    monitor._process = FakeProcess()
    monitor._stop_event = FakeStop()
    monitor._latest = _sample(0, time.monotonic())
    monitor._snapshots_received = 1
    monitor._started = True
    return monitor, connection


def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_receiver_drains_while_no_guard_is_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor, connection = _armed(_output(tmp_path))
    monkeypatch.setattr(C2FResourceMonitor, "start", lambda _self: None)
    monitor.start()
    try:
        for sequence in range(1, 40):
            connection.append(_sample(sequence, time.monotonic()))
        _wait_for(lambda: monitor._snapshots_received == 40)
        assert monitor._latest is not None
        assert monitor._latest["sequence"] == 39
        assert monitor.snapshot()["continuous_receiver"]["thread_alive"] is True
    finally:
        monitor.close()


def test_receiver_failure_is_retained_and_guard_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _output(tmp_path)
    monitor, connection = _armed(output)
    monkeypatch.setattr(C2FResourceMonitor, "start", lambda _self: None)
    monitor.start()
    try:
        connection.error = OSError("pipe failed")
        _wait_for(lambda: monitor._continuous_error is not None)
        allowed, telemetry = monitor.guard(output)
        assert not allowed
        assert "RESOURCE_MONITOR_ERROR" in telemetry["reasons"]
        state = monitor.snapshot()
        assert "OSError: pipe failed" in state["continuous_receiver"]["error"]
        assert state["latest"]["status"] == "ERROR"
    finally:
        monitor.close()


def test_close_stops_receiver_before_parent_resources(tmp_path: Path) -> None:
    monitor, connection = _armed(_output(tmp_path))
    monitor._continuous_thread = None
    monitor.close()
    assert monitor._continuous_stop.is_set()
    assert monitor._closed
    assert connection.closed
    assert monitor._stop_event.set_calls == 1
    assert monitor._process.join_calls == 1


def test_non_object_snapshot_fails_closed(tmp_path: Path) -> None:
    monitor, connection = _armed(_output(tmp_path))
    connection.append("not a snapshot")
    with monitor._lock:
        monitor._drain()
    assert monitor._continuous_error == (
        "RuntimeError: Resource monitor returned a non-object snapshot"
    )
    assert monitor._latest is not None
    assert monitor._latest["status"] == "ERROR"


@pytest.mark.parametrize(
    "change",
    [
        {"other_heavy_processes": None},
        {"inventory_unreadable_python_pids": None},
        {"sampled_monotonic": float("nan")},
        {"sampled_monotonic": float("inf")},
        {"root_pid": 778},
    ],
)
def test_malformed_ok_snapshot_is_sticky_failure(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    monitor, connection = _armed(_output(tmp_path))
    sample = _sample(1, time.monotonic())
    sample.update(change)
    connection.append(sample)
    with monitor._lock:
        monitor._drain()
    assert monitor._continuous_error is not None
    assert monitor._latest is not None
    assert monitor._latest["status"] == "ERROR"


def test_future_sample_clock_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor, connection = _armed(_output(tmp_path))
    connection.append(_sample(1, 101.0))
    monkeypatch.setattr(
        "operational.train40_system.resource_monitor_receiver.time.monotonic", lambda: 100.0
    )
    with monitor._lock:
        monitor._drain()
    assert monitor._continuous_error == (
        "ValueError: Resource monitor sampled_monotonic is in the future"
    )


def test_start_validates_snapshot_installed_by_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor, _connection = _armed(_output(tmp_path))
    assert monitor._latest is not None
    del monitor._latest["disk_free_bytes"]
    monkeypatch.setattr(C2FResourceMonitor, "start", lambda _self: None)
    with pytest.raises(RuntimeError, match="snapshot is invalid"):
        monitor.start()
    assert monitor._closed
    assert monitor._continuous_error is not None


def test_guard_still_uses_original_budget_and_resource_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _output(tmp_path)
    monitor, _connection = _armed(output)
    monkeypatch.setattr(resource_monitor.time, "monotonic", time.monotonic)
    allowed, telemetry = monitor.guard(output)
    assert allowed
    assert telemetry["physical_updates_upper"] == 0
    assert telemetry["reasons"] == []
