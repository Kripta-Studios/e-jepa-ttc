from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.final_refit.shared_resources import (
    GIB,
    PROJECT_RSS_LIMIT_BYTES,
    SharedResourceMonitor,
    SharedResourcePublisher,
    _canonical_bytes,
    _signature,
    acquire_publisher_lock,
    clear_stale_publisher_lock,
    create_shared_resource_session,
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
    command: tuple[str, ...] = ()
    rss: int = 0
    running: bool = True
    process_status: str = "running"
    deny_command: bool = False
    deny_memory: bool = False
    status_calls: int = 0

    def create_time(self) -> float:
        return self.created

    def cmdline(self) -> list[str]:
        if self.deny_command:
            raise AccessDeniedError
        return list(self.command)

    def memory_info(self) -> SimpleNamespace:
        if self.deny_memory:
            raise AccessDeniedError
        return SimpleNamespace(rss=self.rss)

    def is_running(self) -> bool:
        return self.running

    def status(self) -> str:
        self.status_calls += 1
        return self.process_status

    @staticmethod
    def oneshot() -> nullcontext[None]:
        return nullcontext()


class FakeProcessIterator:
    def __init__(self, owner: FakePsutil) -> None:
        self.owner = owner

    def __call__(self, attrs: list[str]) -> list[FakeProcess]:
        assert attrs == ["pid"]
        return list(self.owner.inventory_processes)

    @staticmethod
    def cache_clear() -> None:
        return None


class FakePsutil:
    AccessDenied = AccessDeniedError
    NoSuchProcess = NoSuchProcessError
    ZombieProcess = ZombieProcessError
    STATUS_ZOMBIE = "zombie"

    def __init__(self) -> None:
        self.available = 8 * GIB
        self.disk_free = 30_000_000_000
        self.inventory_processes: list[FakeProcess] = []
        self.processes: dict[int, FakeProcess] = {}
        self.process_iter = FakeProcessIterator(self)

    def Process(self, pid: int) -> FakeProcess:  # noqa: N802
        try:
            return self.processes[pid]
        except KeyError as error:
            raise NoSuchProcessError(pid) from error

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(available=self.available)

    def disk_usage(self, path: str) -> SimpleNamespace:
        del path
        return SimpleNamespace(free=self.disk_free)


@dataclass
class Clock:
    value: int = 1_000_000_000_000

    def now(self) -> int:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += int(seconds * 1_000_000_000)


@dataclass
class Harness:
    repository: Path
    run_root: Path
    session_path: Path
    snapshot_path: Path
    lock_path: Path
    fake: FakePsutil
    clock: Clock
    publisher: SharedResourcePublisher
    monitor: SharedResourceMonitor


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    repository = tmp_path / "repository"
    run_root = tmp_path / "run"
    repository.mkdir()
    run_root.mkdir()
    session_path = run_root / "RESOURCE_SESSION.json"
    snapshot_path = run_root / "RESOURCE_SNAPSHOT.json"
    lock_path = run_root / "RESOURCE_PUBLISHER.lock"
    create_shared_resource_session(
        session_path,
        repository=repository,
        run_root=run_root,
    )
    fake = FakePsutil()
    publisher_process = FakeProcess(os.getpid(), 123.0)
    fake.processes[os.getpid()] = publisher_process
    clock = Clock()
    publisher = SharedResourcePublisher(
        repository,
        run_root,
        session_path=session_path,
        snapshot_path=snapshot_path,
        lock_path=lock_path,
        psutil_module=fake,
        monotonic_ns=clock.now,
        wall_time_ns=clock.now,
    )
    publisher.acquire()
    publisher.publish_once()
    monitor = SharedResourceMonitor(
        repository,
        session_path=session_path,
        snapshot_path=snapshot_path,
        psutil_module=fake,
        commit_reader=lambda: 9 * GIB,
        monotonic_ns=clock.now,
    )
    result = Harness(
        repository,
        run_root,
        session_path,
        snapshot_path,
        lock_path,
        fake,
        clock,
        publisher,
        monitor,
    )
    yield result
    publisher.close()


def _guard(harness: Harness) -> tuple[bool, dict[str, Any]]:
    return harness.monitor.guard(
        harness.run_root,
        checkpoint_reservation=GIB,
        completed_updates=7,
    )


def _rewrite_signed(harness: Harness, **changes: object) -> None:
    snapshot = json.loads(harness.snapshot_path.read_text(encoding="utf-8"))
    snapshot.pop("signature_sha256")
    snapshot.update(changes)
    snapshot["signature_sha256"] = _signature(
        str(harness.monitor.session["session_token"]), snapshot
    )
    harness.snapshot_path.write_bytes(_canonical_bytes(snapshot) + b"\n")


def test_shared_monitor_accepts_bound_snapshot_and_reports_split_counters(
    harness: Harness,
) -> None:
    allowed, values = _guard(harness)
    assert allowed
    assert values["full_process_scan"] is False
    assert values["snapshot_valid"] is True
    assert values["project_tree_rss_upper_bound_bytes"] == 0
    assert values["monitor_statistics"] == {
        "calls": 1,
        "full_scans": 0,
        "guard_seconds_total": values["guard_seconds"],
        "full_guard_seconds_total": 0.0,
        "max_guard_seconds": values["guard_seconds"],
        "client_snapshot_reads": 1,
        "publisher_full_scans_seen": 1,
    }


def test_shared_monitor_accepts_two_gib_commit_boundary(harness: Harness) -> None:
    harness.monitor.commit_reader = lambda: 2 * GIB
    assert _guard(harness)[0] is True


@pytest.mark.parametrize("failure", ["ram", "commit", "disk", "rss", "unknown"])
def test_all_resource_bounds_remain_fail_closed(harness: Harness, failure: str) -> None:
    if failure == "ram":
        harness.fake.available = 2 * GIB - 1
    elif failure == "commit":
        harness.monitor.commit_reader = lambda: 2 * GIB - 1
    elif failure == "disk":
        harness.fake.disk_free = 10_000_000_000 + GIB - 1
    elif failure == "rss":
        _rewrite_signed(
            harness,
            project_tree_rss_upper_bound_bytes=PROJECT_RSS_LIMIT_BYTES + 1,
        )
    else:
        _rewrite_signed(harness, process_coverage_bounded=False)
    assert _guard(harness)[0] is False


@pytest.mark.parametrize("failure", ["missing", "truncated", "stale", "future", "wrong_session"])
def test_invalid_snapshot_is_rejected(harness: Harness, failure: str) -> None:
    if failure == "missing":
        harness.snapshot_path.unlink()
    elif failure == "truncated":
        harness.snapshot_path.write_text('{"schema":', encoding="utf-8")
    elif failure == "stale":
        harness.clock.advance(16.0)
    elif failure == "future":
        _rewrite_signed(
            harness,
            scan_started_monotonic_ns=harness.clock.now() + 1,
            scan_finished_monotonic_ns=harness.clock.now() + 1,
            published_monotonic_ns=harness.clock.now() + 1,
        )
    else:
        _rewrite_signed(harness, session_sha256="0" * 64)
    allowed, values = _guard(harness)
    assert not allowed
    assert values["snapshot_valid"] is False
    assert "snapshot_error" in values


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_tree_rss_bytes", -1),
        ("project_tree_rss_upper_bound_bytes", "23000000000"),
        ("process_coverage_bounded", "false"),
        ("processes_seen", True),
        ("project_processes", "not-a-list"),
    ],
)
def test_authenticated_malformed_inventory_is_denied_without_throwing(
    harness: Harness,
    field: str,
    value: object,
) -> None:
    _rewrite_signed(harness, **{field: value})
    allowed, values = _guard(harness)
    assert not allowed
    assert values["snapshot_valid"] is False
    assert "snapshot_error" in values


def test_authenticated_inventory_consistency_is_enforced(harness: Harness) -> None:
    _rewrite_signed(
        harness,
        process_coverage_bounded=True,
        inspection_failures=[{"pid": 99, "error": "AccessDenied", "rss_upper_bound": -1}],
    )
    allowed, values = _guard(harness)
    assert not allowed
    assert "contradicts inspection failures" in str(values["snapshot_error"])


@pytest.mark.parametrize("failure", ["dead", "pid_reuse"])
def test_publisher_liveness_identity_is_checked_every_call(
    harness: Harness,
    failure: str,
) -> None:
    process = harness.fake.processes[os.getpid()]
    assert _guard(harness)[0]
    if failure == "dead":
        process.running = False
    else:
        process.created = 124.0
    allowed, values = _guard(harness)
    assert not allowed
    assert "publisher PID identity is not live" in str(values["snapshot_error"])


def test_non_windows_zombie_is_rejected(harness: Harness) -> None:
    process = harness.fake.processes[os.getpid()]
    harness.monitor.platform_name = "posix"
    process.process_status = "zombie"
    allowed, values = _guard(harness)
    assert not allowed
    assert process.status_calls == 1
    assert "publisher PID identity is not live" in str(values["snapshot_error"])


def test_windows_suspended_process_does_not_require_thread_enumeration(
    harness: Harness,
) -> None:
    process = harness.fake.processes[os.getpid()]
    harness.monitor.platform_name = "nt"
    process.process_status = "stopped"
    assert _guard(harness)[0]
    assert process.status_calls == 0


def test_atomic_replacement_race_is_denied_then_recovers(harness: Harness) -> None:
    harness.snapshot_path.write_text("{", encoding="utf-8")
    assert _guard(harness)[0] is False
    harness.publisher.publish_once()
    assert _guard(harness)[0] is True


def test_concurrent_publisher_replacement_and_client_reads_are_safe(harness: Harness) -> None:
    barrier = threading.Barrier(2)
    allowed: list[bool] = []
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            barrier.wait()
            for _ in range(30):
                harness.publisher.publish_once()
        except BaseException as error:
            errors.append(error)

    def read() -> None:
        try:
            barrier.wait()
            for _ in range(30):
                allowed.append(_guard(harness)[0])
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=publish), threading.Thread(target=read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert allowed == [True] * 30


def test_publisher_reports_unbounded_inventory_failure(harness: Harness) -> None:
    denied = FakeProcess(
        101,
        11.0,
        deny_command=True,
        deny_memory=True,
    )
    harness.fake.inventory_processes = [denied]
    harness.publisher.publish_once()
    allowed, values = _guard(harness)
    assert not allowed
    assert values["process_coverage_bounded"] is False
    assert values["inspection_failures"][0]["rss_upper_bound"] == -1


def test_project_rss_limit_uses_publisher_inventory(harness: Harness) -> None:
    project = FakeProcess(
        102,
        12.0,
        ("python", "-m", "operational.rgb_port.train_producers"),
        PROJECT_RSS_LIMIT_BYTES + 1,
    )
    harness.fake.inventory_processes = [project]
    harness.publisher.publish_once()
    allowed, values = _guard(harness)
    assert not allowed
    assert values["project_tree_rss_upper_bound_bytes"] == PROJECT_RSS_LIMIT_BYTES + 1


def test_singleton_lock_is_never_stolen_implicitly(harness: Harness) -> None:
    with pytest.raises(FileExistsError):
        acquire_publisher_lock(
            harness.lock_path,
            pid=999,
            create_time=999.0,
            session_sha256=str(harness.publisher.session["descriptor_sha256"]),
        )


def test_stale_lock_cleanup_is_explicit_and_checks_pid_identity(harness: Harness) -> None:
    with pytest.raises(RuntimeError, match="still live"):
        clear_stale_publisher_lock(
            harness.lock_path,
            session_path=harness.session_path,
            repository=harness.repository,
            run_root=harness.run_root,
            psutil_module=harness.fake,
        )
    harness.fake.processes[os.getpid()].created = 999.0
    clear_stale_publisher_lock(
        harness.lock_path,
        session_path=harness.session_path,
        repository=harness.repository,
        run_root=harness.run_root,
        psutil_module=harness.fake,
    )
    assert not harness.lock_path.exists()
    harness.publisher.lock = None


def test_session_creation_and_guard_run_root_are_exactly_bound(harness: Harness) -> None:
    with pytest.raises(FileExistsError):
        create_shared_resource_session(
            harness.session_path,
            repository=harness.repository,
            run_root=harness.run_root,
        )
    other = harness.run_root.parent / "other-run"
    other.mkdir()
    allowed, values = harness.monitor.guard(
        other,
        checkpoint_reservation=GIB,
        completed_updates=7,
    )
    assert not allowed
    assert values["run_root_binding_valid"] is False


def test_corrupt_session_is_rejected_before_client_starts(harness: Harness) -> None:
    session = json.loads(harness.session_path.read_text(encoding="utf-8"))
    session["run_root"] = str(harness.run_root / "other")
    harness.session_path.write_text(json.dumps(session), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        SharedResourceMonitor(
            harness.repository,
            session_path=harness.session_path,
            snapshot_path=harness.snapshot_path,
            psutil_module=harness.fake,
        )
