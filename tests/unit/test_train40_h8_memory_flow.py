"""Tests for H8 RAM backpressure that never skips scientific work."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.train40_system import h8_memory_flow as flow
from operational.train40_system.resource_monitor_receiver import ContinuousC2FResourceMonitor


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _telemetry(
    *, host: int, tree: int = 10_000_000_000, reasons: list[str] | None = None, sequence: int = 1
) -> dict[str, Any]:
    return {
        "tree_rss_bytes": tree,
        "host_available_bytes": host,
        "monitor_sequence": sequence,
        "reasons": [] if reasons is None else reasons,
    }


def _guard_sequence(
    values: list[tuple[bool, dict[str, Any]]],
) -> Iterator[tuple[bool, dict[str, Any]]]:
    yield from values


def _monitor(
    tmp_path: Path, fake: FakeTime, host_values: list[int]
) -> flow.MemoryFlowMonitor:
    values = iter(host_values)
    return flow.MemoryFlowMonitor(
        tmp_path,
        clock=fake.clock,
        sleep=fake.sleep,
        host_available=lambda: next(values),
    )


def test_low_host_waits_until_three_gib_hysteresis(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [1_000, 2_500_000_000, 4_000_000_000])
    samples = _guard_sequence(
        [
            (False, _telemetry(host=1_000, reasons=["RAM_RESOURCE_BOUNDARY"])),
            (True, _telemetry(host=2_500_000_000, sequence=2)),
            (True, _telemetry(host=4_000_000_000, sequence=3)),
        ]
    )
    publications: list[dict[str, Any]] = []
    with (
        patch.object(
            ContinuousC2FResourceMonitor, "guard", side_effect=lambda _path: next(samples)
        ),
        patch.object(
            flow, "atomic_json", side_effect=lambda _path, value: publications.append(value)
        ),
    ):
        allowed, telemetry = monitor.guard(tmp_path)
    assert allowed
    assert fake.sleeps == [1.0, 1.0]
    assert telemetry["memory_flow_state"] == "NORMAL"
    assert monitor._flow_wait_seconds == 2.0
    assert publications[0]["status"] == "WAITING_FOR_HOST_MEMORY"
    assert publications[-1]["status"] == "NORMAL"
    assert all(item["optimizer_updates"] == 0 for item in publications)
    assert all(item["scientific_negative"] is False for item in publications)


def test_stop_request_aborts_while_waiting(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [1_000, 1_000])
    samples = _guard_sequence(
        [
            (False, _telemetry(host=1_000, reasons=["RAM_RESOURCE_BOUNDARY"])),
            (
                False,
                _telemetry(
                    host=1_000,
                    reasons=["RAM_RESOURCE_BOUNDARY", "COORDINATION_PAUSE_REQUESTED"],
                    sequence=2,
                ),
            ),
        ]
    )
    with (
        patch.object(
            ContinuousC2FResourceMonitor, "guard", side_effect=lambda _path: next(samples)
        ),
        patch.object(flow, "atomic_json"),
    ):
        allowed, telemetry = monitor.guard(tmp_path)
    assert not allowed
    assert "COORDINATION_PAUSE_REQUESTED" in telemetry["reasons"]
    assert fake.sleeps == [1.0]


def test_non_ram_reason_is_never_suppressed(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [5_000_000_000])
    telemetry = _telemetry(host=5_000_000_000, reasons=["CHECKPOINT_DISK_RESERVE"])
    with (
        patch.object(ContinuousC2FResourceMonitor, "guard", return_value=(False, telemetry)),
        patch.object(flow, "atomic_json"),
    ):
        allowed, result = monitor.guard(tmp_path)
    assert not allowed
    assert result["reasons"] == ["CHECKPOINT_DISK_RESERVE"]
    assert fake.sleeps == []


def test_first_normal_guard_publishes_flow_status(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [5_000_000_000])
    telemetry = _telemetry(host=4_500_000_000)
    publications: list[dict[str, Any]] = []
    with (
        patch.object(ContinuousC2FResourceMonitor, "guard", return_value=(True, telemetry)),
        patch.object(
            flow, "atomic_json", side_effect=lambda _path, value: publications.append(value)
        ),
    ):
        allowed, result = monitor.guard(tmp_path)
    assert allowed
    assert result["host_available_source"] == "fresh_psutil"
    assert publications[0]["status"] == "NORMAL"
    assert publications[0]["checked_utc"]


def test_aggregate_rss_only_throttles_and_admits(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [4_000_000_000])
    telemetry = _telemetry(
        host=4_000_000_000,
        tree=23_100_000_000,
        reasons=["RAM_RESOURCE_BOUNDARY"],
    )
    with (
        patch.object(ContinuousC2FResourceMonitor, "guard", return_value=(False, telemetry)),
        patch.object(flow, "atomic_json"),
    ):
        allowed, result = monitor.guard(tmp_path)
    assert allowed
    assert result["reasons"] == []
    assert result["memory_pressure"] is True
    assert fake.sleeps == [0.5]
    assert monitor.snapshot()["memory_flow"]["throttle_count"] == 1


def test_wait_publication_repeats_every_ten_seconds(tmp_path: Path) -> None:
    fake = FakeTime()
    monitor = _monitor(tmp_path, fake, [1_000] * 12 + [4_000_000_000])
    low = _telemetry(host=1_000, reasons=["RAM_RESOURCE_BOUNDARY"])
    values = [(False, {**low, "monitor_sequence": value}) for value in range(12)]
    values.append((True, _telemetry(host=4_000_000_000, sequence=12)))
    samples = _guard_sequence(values)
    publications: list[dict[str, Any]] = []
    with (
        patch.object(
            ContinuousC2FResourceMonitor, "guard", side_effect=lambda _path: next(samples)
        ),
        patch.object(
            flow, "atomic_json", side_effect=lambda _path, value: publications.append(value)
        ),
    ):
        allowed, _ = monitor.guard(tmp_path)
    assert allowed
    waiting = [item for item in publications if item["status"] == "WAITING_FOR_HOST_MEMORY"]
    assert len(waiting) >= 2
    assert waiting[-1]["wait_seconds"] >= 10.0


def test_fresh_import_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import operational.train40_system.h8_memory_flow; "
                "raise SystemExit(1 if 'torch' in sys.modules else 0)"
            ),
        ],
        cwd=Path.cwd(),
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
