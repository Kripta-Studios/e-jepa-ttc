"""CPU-only tests for the independent TRAIN40 resource monitor."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.train40_system import resource_monitor
from operational.train40_system.resource_monitor import ResourceMonitor


class AccessDeniedError(Exception):
    pass


class NoSuchProcessError(Exception):
    pass


class ZombieProcessError(Exception):
    pass


@dataclass
class FakeProcess:
    pid: int
    created: float
    name: str = "python.exe"
    command: tuple[str, ...] = ()
    rss: int = 100
    parent_values: tuple[FakeProcess, ...] = ()
    child_values: tuple[FakeProcess, ...] = ()
    inaccessible: bool = False

    @property
    def info(self) -> dict[str, str]:
        return {"name": self.name}

    def create_time(self) -> float:
        if self.inaccessible:
            raise AccessDeniedError
        return self.created

    def cmdline(self) -> list[str]:
        if self.inaccessible:
            raise AccessDeniedError
        return list(self.command)

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss)

    def parents(self) -> list[FakeProcess]:
        return list(self.parent_values)

    def children(self, *, recursive: bool) -> list[FakeProcess]:
        assert recursive
        return list(self.child_values)


class FakePsutil:
    AccessDenied = AccessDeniedError
    NoSuchProcess = NoSuchProcessError
    ZombieProcess = ZombieProcessError

    def __init__(
        self,
        root: FakeProcess,
        processes: list[FakeProcess],
        fresh: dict[int, FakeProcess] | None = None,
    ) -> None:
        self.root = root
        self.processes = processes
        self.fresh = fresh or {process.pid: process for process in processes}
        self.fresh[root.pid] = root

    def Process(self, pid: int) -> FakeProcess:  # noqa: N802
        return self.fresh[pid]

    def process_iter(self, attrs: list[str]) -> list[FakeProcess]:
        assert attrs == ["name"]
        return self.processes

    @staticmethod
    def virtual_memory() -> SimpleNamespace:
        return SimpleNamespace(available=9_000)


def _output(tmp_path: Path, *, previous: int = 100, synthetic: int = 5) -> Path:
    (tmp_path / "fits/a5_seed7").mkdir(parents=True)
    (tmp_path / "AUTHORIZATION.json").write_text(
        json.dumps(
            {
                "previous_physical_execution_upper": previous,
                "deadline_utc": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "TECHNICAL_ACCOUNTING.json").write_text(
        json.dumps({"synthetic_optimizer_updates": synthetic}), encoding="utf-8"
    )
    (tmp_path / "fits/a5_seed7/UPDATE_JOURNAL.json").write_text(
        json.dumps(
            {
                "committed_updates": 10,
                "durable_updates": 9,
                "recovery_upper": 2,
                "pending_update_upper": 1,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _sample(now: float = 100.0) -> dict[str, Any]:
    return {
        "status": "OK",
        "sequence": 3,
        "sampled_monotonic": now,
        "tree_rss_bytes": 1_000,
        "host_available_bytes": 9_000_000_000,
        "disk_free_bytes": 30_000_000_000,
        "other_heavy_processes": [],
        "inventory_unreadable_python_pids": [],
    }


class AliveProcess:
    pid = 123

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


class OneShotStop:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, timeout: float) -> bool:
        del timeout
        self.stopped = True
        return True


class CapturingSender:
    def __init__(self) -> None:
        self.values: list[object] = []

    def send(self, value: object) -> None:
        self.values.append(value)

    def close(self) -> None:
        return


def _armed(output: Path, sample: dict[str, Any], *, alive: bool = True) -> ResourceMonitor:
    monitor = ResourceMonitor(output)
    monitor._started = True
    monitor._latest = sample
    monitor._process = AliveProcess(alive)
    return monitor


def test_collect_detects_new_heavy_after_related_pid_reuse(tmp_path: Path) -> None:
    parent = FakeProcess(1, 1.0)
    old_child = FakeProcess(11, 2.0)
    root = FakeProcess(10, 1.5, parent_values=(parent,), child_values=(old_child,))
    reused = FakeProcess(
        11,
        9.0,
        command=("python", "-m", "operational.train40_system.engine_other"),
        rss=500,
    )
    non_python = FakeProcess(30, 3.0, name="browser.exe", command=("private",))
    fake = FakePsutil(root, [root, reused, non_python], fresh={11: reused})

    sample = resource_monitor._collect_sample(
        10,
        1.5,
        tmp_path,
        psutil_module=fake,
        disk_usage=lambda _path: SimpleNamespace(free=50_000),
        clock=lambda: 22.0,
    )

    assert sample["tree_rss_bytes"] == 100
    assert sample["other_heavy_processes"] == [{"pid": 11, "create_time": 9.0}]
    assert sample["python_cmdline_reads"] == 2


def test_collect_reports_unreadable_python_inventory_fail_closed(tmp_path: Path) -> None:
    root = FakeProcess(10, 1.5)
    hidden = FakeProcess(20, 2.0, inaccessible=True)
    fake = FakePsutil(root, [root, hidden])
    sample = resource_monitor._collect_sample(
        10,
        1.5,
        tmp_path,
        psutil_module=fake,
        disk_usage=lambda _path: SimpleNamespace(free=50_000),
        clock=lambda: 22.0,
    )
    assert sample["inventory_unreadable_python_pids"] == [20]


def test_guard_admits_fresh_safe_snapshot_and_preserves_budget_math(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _output(tmp_path)
    monitor = _armed(output, _sample())
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: 105.0)
    permitted, telemetry = monitor.guard(output)
    assert permitted
    assert telemetry["physical_updates_upper"] == 118
    assert telemetry["new_recovery_upper"] == 2
    assert telemetry["pending_update_upper"] == 1
    assert telemetry["reasons"] == []
    runtime = monitor.snapshot()
    assert runtime["guard_calls"] == 1
    assert runtime["guard_total_seconds"] >= runtime["last_guard_seconds"] >= 0.0


def test_worker_fails_closed_if_spawn_entrypoint_imported_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sender = CapturingSender()
    monkeypatch.setitem(resource_monitor.sys.modules, "torch", object())
    resource_monitor._worker(1, 1.0, str(tmp_path), sender, OneShotStop())
    assert len(sender.values) == 1
    sample = sender.values[0]
    assert isinstance(sample, dict)
    assert sample["status"] == "ERROR"
    assert sample["torch_imported"] is True
    assert "Torch imported" in str(sample["error"])


def test_production_spawn_entrypoint_keeps_heavy_imports_lazy() -> None:
    source = Path("operational/train40_system/engine_host_pipeline.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    top_imports: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            top_imports.add(statement.module.split(".")[0])
    assert top_imports <= {"__future__", "argparse", "pathlib", "typing"}
    assert 'if __name__ == "__main__":' in source


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"sampled_monotonic": 80.0}, "RESOURCE_MONITOR_STALE"),
        ({"tree_rss_bytes": resource_monitor.TREE_RSS_LIMIT_BYTES}, "RAM_RESOURCE_BOUNDARY"),
        ({"host_available_bytes": 1}, "RAM_RESOURCE_BOUNDARY"),
        ({"disk_free_bytes": 1}, "CHECKPOINT_DISK_RESERVE"),
        (
            {"other_heavy_processes": [{"pid": 44, "create_time": 3.0}]},
            "ANOTHER_HEAVY_TRAINER",
        ),
        ({"inventory_unreadable_python_pids": [55]}, "RESOURCE_MONITOR_INVENTORY_INCOMPLETE"),
    ],
)
def test_guard_fails_closed_for_monitor_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: dict[str, Any],
    reason: str,
) -> None:
    output = _output(tmp_path)
    sample = _sample()
    sample.update(change)
    monitor = _armed(output, sample)
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: 100.0)
    permitted, telemetry = monitor.guard(output)
    assert not permitted
    assert reason in telemetry["reasons"]


def test_guard_fails_closed_when_monitor_process_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _output(tmp_path)
    monitor = _armed(output, _sample(), alive=False)
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: 100.0)
    permitted, telemetry = monitor.guard(output)
    assert not permitted
    assert "RESOURCE_MONITOR_PROCESS_DEAD" in telemetry["reasons"]


def test_guard_enforces_physical_recovery_deadline_and_pause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _output(tmp_path, previous=resource_monitor.PHYSICAL_UPDATE_LIMIT)
    journal = output / "fits/a5_seed7/UPDATE_JOURNAL.json"
    value = json.loads(journal.read_text(encoding="utf-8"))
    value["recovery_upper"] = resource_monitor.RECOVERY_LIMIT
    journal.write_text(json.dumps(value), encoding="utf-8")
    authorization = output / "AUTHORIZATION.json"
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["deadline_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    authorization.write_text(json.dumps(value), encoding="utf-8")
    (output / "STOP_REQUEST").write_text("stop", encoding="utf-8")
    monitor = _armed(output, _sample())
    monkeypatch.setattr(resource_monitor.time, "monotonic", lambda: 100.0)
    permitted, telemetry = monitor.guard(output)
    assert not permitted
    assert {
        "PHYSICAL_UPDATE_CAP",
        "NEW_RECOVERY_RESERVE_EXHAUSTED",
        "USER_DEADLINE",
        "COORDINATION_PAUSE_REQUESTED",
    }.issubset(telemetry["reasons"])


def test_guard_rejects_different_output(tmp_path: Path) -> None:
    output = _output(tmp_path / "one")
    monitor = _armed(output, _sample())
    with pytest.raises(ValueError, match="differs"):
        monitor.guard(tmp_path / "two")


def test_real_spawn_publishes_small_torch_free_snapshot(tmp_path: Path) -> None:
    output = _output(tmp_path)
    monitor = ResourceMonitor(output)
    try:
        monitor.start()
        state = monitor.snapshot()
        assert state["started"]
        assert state["snapshots_received"] >= 1
        assert state["latest"]["status"] == "OK"
        assert state["latest"]["torch_imported"] is False
    finally:
        monitor.close()
