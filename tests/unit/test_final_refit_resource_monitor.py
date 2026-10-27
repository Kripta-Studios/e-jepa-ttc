from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.final_refit.resource_monitor import (
    GIB,
    CachedProcessInventory,
    ResourceMonitor,
    assert_recovery_budget,
    recovery_upper,
)


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
    command: tuple[str, ...]
    rss: int
    denied: bool = False
    cmdline_calls: int = 0

    def create_time(self) -> float:
        if self.denied:
            raise AccessDeniedError
        return self.created

    def cmdline(self) -> list[str]:
        if self.denied:
            raise AccessDeniedError
        return list(self.command)

    def memory_info(self) -> SimpleNamespace:
        if self.denied:
            raise AccessDeniedError
        return SimpleNamespace(rss=self.rss)

    @staticmethod
    def oneshot() -> nullcontext[None]:
        return nullcontext()


class FakePsutil:
    AccessDenied = AccessDeniedError
    NoSuchProcess = NoSuchProcessError
    ZombieProcess = ZombieProcessError

    def __init__(self, processes: list[FakeProcess]) -> None:
        self.processes = processes
        self.process_iter = FakeProcessIterator(self)

    @staticmethod
    def virtual_memory() -> SimpleNamespace:
        return SimpleNamespace(available=8 * GIB)

    @staticmethod
    def disk_usage(path: str) -> SimpleNamespace:
        del path
        return SimpleNamespace(free=30_000_000_000)


class CmdlineDeniedProcess(FakeProcess):
    def cmdline(self) -> list[str]:
        self.cmdline_calls += 1
        raise AccessDeniedError


class FakeProcessIterator:
    def __init__(self, owner: FakePsutil) -> None:
        self.owner = owner
        self.cache_clear_calls = 0

    def __call__(self, attrs: list[str]) -> list[FakeProcess]:
        assert attrs == ["pid"]
        return self.owner.processes

    def cache_clear(self) -> None:
        self.cache_clear_calls += 1


def test_inventory_caches_cmdline_but_refreshes_rss_and_detects_exit_reuse(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    process = FakeProcess(10, 1.0, ("python", "repo/train.py"), 100)
    fake = FakePsutil([process])
    inventory = CachedProcessInventory(repository, psutil_module=fake)
    first = inventory.scan()
    assert (first.cmdline_reads, first.cmdline_cache_hits) == (1, 0)
    assert first.project_tree_rss_bytes == 100
    process.rss = 250
    second = inventory.scan()
    assert (second.cmdline_reads, second.cmdline_cache_hits) == (0, 1)
    assert second.project_tree_rss_bytes == 250
    fake.processes = [FakeProcess(10, 2.0, ("python", "unrelated.py"), 999)]
    reused = inventory.scan()
    assert reused.reused_pids == (10,)
    assert reused.exited_identities == 1
    assert reused.project_processes == ()
    fake.processes = []
    exited = inventory.scan()
    assert exited.exited_identities == 1
    assert fake.process_iter.cache_clear_calls == 4


def test_inventory_failure_is_reported_for_fail_closed_guard(tmp_path: Path) -> None:
    denied = FakeProcess(20, 2.0, ("python",), 1, denied=True)
    fake = FakePsutil([denied])
    monitor = ResourceMonitor(tmp_path, psutil_module=fake, commit_reader=lambda: 9 * GIB)
    allowed, values = monitor.guard(tmp_path, checkpoint_reservation=1 * GIB, completed_updates=0)
    assert not allowed
    assert values["inspection_failures"] == [
        {"pid": 20, "error": "AccessDeniedError", "rss_upper_bound": -1}
    ]


def test_cached_identity_does_not_hide_later_inspection_failure(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    process = FakeProcess(20, 2.0, ("python", str(repository)), 100)
    fake = FakePsutil([process])
    monitor = ResourceMonitor(repository, psutil_module=fake, commit_reader=lambda: 9 * GIB)
    assert monitor.guard(repository, checkpoint_reservation=GIB, completed_updates=0)[0]
    process.denied = True
    allowed, values = monitor.guard(repository, checkpoint_reservation=GIB, completed_updates=25)
    assert not allowed
    assert values["inspection_failures"] == [
        {"pid": 20, "error": "AccessDeniedError", "rss_upper_bound": -1}
    ]


def test_inventory_preserves_pytest_exclusion(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    process = FakeProcess(22, 2.0, ("python", "-m", "pytest", str(repository)), 100)
    inventory = CachedProcessInventory(repository, psutil_module=FakePsutil([process])).scan()
    assert inventory.project_processes == ()


def test_unclassified_process_rss_is_a_conservative_upper_bound(tmp_path: Path) -> None:
    process = CmdlineDeniedProcess(23, 2.0, (), 1234)
    fake = FakePsutil([process])
    monitor = ResourceMonitor(tmp_path, psutil_module=fake, commit_reader=lambda: 9 * GIB)
    allowed, values = monitor.guard(tmp_path, checkpoint_reservation=GIB, completed_updates=0)
    assert allowed
    assert values["project_tree_rss_bytes"] == 0
    assert values["project_tree_rss_upper_bound_bytes"] == 1234
    assert values["process_coverage_bounded"] is True
    assert values["inspection_failures"] == [
        {"pid": 23, "error": "AccessDeniedError", "rss_upper_bound": 1234}
    ]
    allowed, values = monitor.guard(tmp_path, checkpoint_reservation=GIB, completed_updates=25)
    assert allowed
    assert process.cmdline_calls == 1
    assert values["inspection_failures"] == [
        {"pid": 23, "error": "CachedAccessDenied", "rss_upper_bound": 1234}
    ]


@pytest.mark.parametrize("headroom,expected", [(2 * GIB - 1, False), (2 * GIB, True)])
def test_commit_boundary_is_two_gib(tmp_path: Path, headroom: int, expected: bool) -> None:
    monitor = ResourceMonitor(
        tmp_path, psutil_module=FakePsutil([]), commit_reader=lambda: headroom
    )
    allowed, _ = monitor.guard(tmp_path, checkpoint_reservation=GIB, completed_updates=0)
    assert allowed is expected


def test_full_scan_cadence_is_first_and_every_25_updates(tmp_path: Path) -> None:
    fake = FakePsutil([])
    monitor = ResourceMonitor(tmp_path, psutil_module=fake, commit_reader=lambda: 9 * GIB)
    results = [
        monitor.guard(tmp_path, checkpoint_reservation=1 * GIB, completed_updates=update)[1][
            "full_process_scan"
        ]
        for update in (0, 1, 24, 25, 49, 50)
    ]
    assert results == [True, False, False, True, False, True]
    statistics = monitor.statistics()
    assert statistics["calls"] == 6
    assert statistics["full_scans"] == 3
    assert float(statistics["guard_seconds_total"]) >= float(statistics["full_guard_seconds_total"])


def test_recovery_upper_includes_completed_durable_gap(tmp_path: Path) -> None:
    fit = tmp_path / "fits/R_A5"
    fit.mkdir(parents=True)
    (fit / "UPDATE_JOURNAL.json").write_text(
        json.dumps(
            {
                "completed_updates": 169,
                "durable_updates": 100,
                "recovery_upper": 3,
                "pending_update_upper": 1,
            }
        ),
        encoding="utf-8",
    )
    assert recovery_upper(tmp_path) == 73
    assert_recovery_budget(tmp_path, prospective_pending=1, limit=74)
    with pytest.raises(RuntimeError, match="75 > 74"):
        assert_recovery_budget(tmp_path, prospective_pending=2, limit=74)
