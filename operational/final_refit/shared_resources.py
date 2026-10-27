"""Shared, authenticated process inventories for concurrent final-refit trainers.

The publisher is an explicit foreground process.  It performs the expensive
``CachedProcessInventory`` scan and atomically publishes an HMAC-authenticated
snapshot.  Trainer clients retain fresh RAM, commit and disk checks on every
update, and fail closed when the publisher or its snapshot cannot be verified.

Nothing in this module starts a supervisor or falls back to an in-process full
scan.  Launcher code owns the publisher lifecycle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import time
from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import psutil

from operational.final_refit.resource_monitor import (
    GIB,
    CachedProcessInventory,
    ProcessInventory,
    _windows_commit_headroom,
)
from operational.rgb_port.accounting import (
    atomic_write_json as _accounting_atomic_write_json,
)
from operational.rgb_port.accounting import read_json_shared as _accounting_read_json_shared

SESSION_SCHEMA = "final_refit_shared_resource_session_v1"
SNAPSHOT_SCHEMA = "final_refit_shared_resource_snapshot_v1"
LOCK_SCHEMA = "final_refit_shared_resource_publisher_lock_v1"
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_TTL_SECONDS = 15.0
PROJECT_RSS_LIMIT_BYTES = 23_000_000_000


class _MemoryInfo(Protocol):
    rss: int


class _VirtualMemory(Protocol):
    available: int


class _DiskUsage(Protocol):
    free: int


class _PublisherProcess(Protocol):
    pid: int

    def create_time(self) -> float: ...

    def is_running(self) -> bool: ...

    def status(self) -> str: ...


class _Psutil(Protocol):
    AccessDenied: type[BaseException]
    NoSuchProcess: type[BaseException]
    ZombieProcess: type[BaseException]
    STATUS_ZOMBIE: str

    def Process(self, pid: int) -> _PublisherProcess: ...  # noqa: N802

    def virtual_memory(self) -> _VirtualMemory: ...

    def disk_usage(self, path: str) -> _DiskUsage: ...


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signature(token: str, payload: Mapping[str, Any]) -> str:
    return hmac.new(bytes.fromhex(token), _canonical_bytes(payload), hashlib.sha256).hexdigest()


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"snapshot {name} must be an integer >= {minimum}")
    return value


def _strict_finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"snapshot {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"snapshot {name} must be a finite positive number")
    return number


def _validate_inventory_payload(snapshot: Mapping[str, Any]) -> None:
    """Reject authenticated but malformed or internally inconsistent inventories."""
    project_rss = _strict_int(snapshot.get("project_tree_rss_bytes"), "project RSS")
    upper_rss = _strict_int(
        snapshot.get("project_tree_rss_upper_bound_bytes"),
        "project RSS upper bound",
    )
    if upper_rss < project_rss:
        raise ValueError("snapshot project RSS upper bound is below measured RSS")
    for field in (
        "processes_seen",
        "cmdline_reads",
        "cmdline_cache_hits",
        "exited_identities",
    ):
        _strict_int(snapshot.get(field), field)

    processes = snapshot.get("project_processes")
    if not isinstance(processes, list):
        raise ValueError("snapshot project_processes must be a list")
    identities: set[tuple[int, float]] = set()
    process_rss_sum = 0
    for item in processes:
        if not isinstance(item, dict):
            raise ValueError("snapshot project process must be an object")
        pid = _strict_int(item.get("pid"), "project process pid", minimum=1)
        created = _strict_finite_number(
            item.get("create_time"),
            "project process create_time",
            positive=True,
        )
        rss = _strict_int(item.get("rss"), "project process RSS")
        identity = (pid, created)
        if identity in identities:
            raise ValueError("snapshot contains a duplicate project process identity")
        identities.add(identity)
        process_rss_sum += rss
    if process_rss_sum != project_rss:
        raise ValueError("snapshot project process RSS does not sum to measured RSS")

    reused = snapshot.get("reused_pids")
    if not isinstance(reused, list):
        raise ValueError("snapshot reused_pids must be a list")
    reused_values = [_strict_int(value, "reused pid", minimum=1) for value in reused]
    if reused_values != sorted(set(reused_values)):
        raise ValueError("snapshot reused_pids must be sorted and unique")

    failures = snapshot.get("inspection_failures")
    if not isinstance(failures, list):
        raise ValueError("snapshot inspection_failures must be a list")
    bounded_unknown_rss = 0
    has_unbounded_failure = False
    for item in failures:
        if not isinstance(item, dict):
            raise ValueError("snapshot inspection failure must be an object")
        _strict_int(item.get("pid"), "inspection failure pid", minimum=1)
        if not isinstance(item.get("error"), str) or not item["error"]:
            raise ValueError("snapshot inspection failure error must be a non-empty string")
        failure_upper = _strict_int(
            item.get("rss_upper_bound"),
            "inspection failure RSS upper bound",
            minimum=-1,
        )
        if failure_upper < 0:
            has_unbounded_failure = True
        else:
            bounded_unknown_rss += failure_upper
    bounded = snapshot.get("process_coverage_bounded")
    if type(bounded) is not bool:
        raise ValueError("snapshot process_coverage_bounded must be a bool")
    if bounded == has_unbounded_failure:
        raise ValueError("snapshot process coverage contradicts inspection failures")
    if upper_rss != project_rss + bounded_unknown_rss:
        raise ValueError("snapshot project RSS upper bound contradicts inspection failures")

    scan_error = snapshot.get("publisher_scan_error")
    if scan_error is not None and (not isinstance(scan_error, str) or not scan_error):
        raise ValueError("snapshot publisher_scan_error must be null or a non-empty string")
    if scan_error is not None and bounded:
        raise ValueError("snapshot scan error cannot claim bounded process coverage")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Use the audited Windows-sharing/retry writer from RGB-PORT accounting."""
    _accounting_atomic_write_json(path, value)


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a JSON contract exactly once without a check/create race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    return _accounting_read_json_shared(path)


def create_shared_resource_session(
    session_path: Path,
    *,
    repository: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Create a new immutable session descriptor with a random HMAC key."""
    payload: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "repository": str(repository.resolve()),
        "run_root": str(run_root.resolve()),
        "session_token": secrets.token_hex(32),
        "created_wall_time_ns": time.time_ns(),
    }
    descriptor = {**payload, "descriptor_sha256": _sha256(_canonical_bytes(payload))}
    _exclusive_write_json(session_path, descriptor)
    return descriptor


def load_shared_resource_session(
    session_path: Path,
    *,
    repository: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Load and bind a session to the exact repository and run root."""
    descriptor = _read_json(session_path)
    digest = str(descriptor.pop("descriptor_sha256", ""))
    if descriptor.get("schema") != SESSION_SCHEMA:
        raise ValueError("Unsupported shared-resource session schema")
    if not hmac.compare_digest(digest, _sha256(_canonical_bytes(descriptor))):
        raise ValueError("Shared-resource session digest mismatch")
    if Path(str(descriptor.get("repository", ""))).resolve() != repository.resolve():
        raise ValueError("Shared-resource session repository mismatch")
    if Path(str(descriptor.get("run_root", ""))).resolve() != run_root.resolve():
        raise ValueError("Shared-resource session run-root mismatch")
    token = str(descriptor.get("session_token", ""))
    if len(token) != 64:
        raise ValueError("Shared-resource session token is invalid")
    try:
        bytes.fromhex(token)
    except ValueError as error:
        raise ValueError("Shared-resource session token is invalid") from error
    return {**descriptor, "descriptor_sha256": digest}


@dataclass(frozen=True)
class PublisherLock:
    """Exclusive publisher lock owned by one PID identity."""

    path: Path
    pid: int
    create_time: float
    session_sha256: str

    def release(self) -> None:
        """Remove the lock only when it still contains this exact identity."""
        try:
            current = _read_json(self.path)
        except FileNotFoundError:
            return
        expected = {
            "schema": LOCK_SCHEMA,
            "pid": self.pid,
            "create_time": self.create_time,
            "session_sha256": self.session_sha256,
        }
        if current != expected:
            raise RuntimeError("Publisher lock identity changed; refusing to remove it")
        self.path.unlink()


def acquire_publisher_lock(
    lock_path: Path,
    *,
    pid: int,
    create_time: float,
    session_sha256: str,
) -> PublisherLock:
    """Acquire a local-run singleton without automatically stealing stale locks."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LOCK_SCHEMA,
        "pid": int(pid),
        "create_time": float(create_time),
        "session_sha256": session_sha256,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return PublisherLock(lock_path, int(pid), float(create_time), session_sha256)


def clear_stale_publisher_lock(
    lock_path: Path,
    *,
    session_path: Path,
    repository: Path,
    run_root: Path,
    psutil_module: object = psutil,
) -> None:
    """Explicitly clear a lock only after proving its exact publisher is gone."""
    session = load_shared_resource_session(
        session_path,
        repository=repository,
        run_root=run_root,
    )
    lock = _read_json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Publisher lock schema mismatch")
    if lock.get("session_sha256") != session["descriptor_sha256"]:
        raise ValueError("Publisher lock session mismatch")
    pid = int(lock["pid"])
    created = float(lock["create_time"])
    module = cast(_Psutil, psutil_module)
    try:
        process = module.Process(pid)
        same_live_identity = bool(
            process.is_running()
            and process.status() != module.STATUS_ZOMBIE
            and float(process.create_time()) == created
        )
    except (module.NoSuchProcess, module.ZombieProcess):
        same_live_identity = False
    except (module.AccessDenied, OSError) as error:
        raise RuntimeError("Cannot prove that the publisher lock is stale") from error
    if same_live_identity:
        raise RuntimeError("Publisher is still live; refusing to clear its lock")
    if _read_json(lock_path) != lock:
        raise RuntimeError("Publisher lock changed during stale-lock validation")
    lock_path.unlink()


def _inventory_payload(inventory: ProcessInventory) -> dict[str, Any]:
    failures = [dict(item) for item in inventory.inspection_failures]
    return {
        "project_tree_rss_bytes": inventory.project_tree_rss_bytes,
        "project_tree_rss_upper_bound_bytes": inventory.project_tree_rss_upper_bound_bytes,
        "project_processes": [dict(item) for item in inventory.project_processes],
        "processes_seen": inventory.processes_seen,
        "cmdline_reads": inventory.cmdline_reads,
        "cmdline_cache_hits": inventory.cmdline_cache_hits,
        "exited_identities": inventory.exited_identities,
        "reused_pids": list(inventory.reused_pids),
        "inspection_failures": failures,
        "process_coverage_bounded": not any(
            int(item.get("rss_upper_bound", -1)) < 0 for item in failures
        ),
    }


class SharedResourcePublisher:
    """Perform expensive inventories and publish authenticated snapshots."""

    def __init__(
        self,
        repository: Path,
        run_root: Path,
        *,
        session_path: Path,
        snapshot_path: Path,
        lock_path: Path,
        psutil_module: object = psutil,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.repository = repository.resolve()
        self.run_root = run_root.resolve()
        self.repository_text = str(self.repository)
        self.run_root_text = str(self.run_root)
        self.run_root_key = os.path.normcase(os.path.abspath(self.run_root))
        self.disk_anchor = self.run_root.anchor
        self.session = load_shared_resource_session(
            session_path,
            repository=self.repository,
            run_root=self.run_root,
        )
        self.snapshot_path = snapshot_path
        self.lock_path = lock_path
        self.module = cast(_Psutil, psutil_module)
        self.monotonic_ns = monotonic_ns
        self.wall_time_ns = wall_time_ns
        self.inventory = CachedProcessInventory(repository, psutil_module=psutil_module)
        self.pid = os.getpid()
        self.process = self.module.Process(self.pid)
        self.create_time = float(self.process.create_time())
        self.sequence = 0
        self.lock: PublisherLock | None = None

    def acquire(self) -> None:
        """Acquire this run's publisher singleton."""
        if self.lock is not None:
            raise RuntimeError("Publisher lock is already held")
        self.lock = acquire_publisher_lock(
            self.lock_path,
            pid=self.pid,
            create_time=self.create_time,
            session_sha256=str(self.session["descriptor_sha256"]),
        )

    def close(self) -> None:
        """Release the singleton on an orderly launcher shutdown."""
        if self.lock is not None:
            self.lock.release()
            self.lock = None

    def publish_once(self) -> dict[str, Any]:
        """Scan and atomically publish one snapshot."""
        if self.lock is None:
            raise RuntimeError("Publisher lock has not been acquired")
        scan_started = self.monotonic_ns()
        try:
            inventory = _inventory_payload(self.inventory.scan())
            scan_error: str | None = None
        except Exception as error:  # publisher stays alive but clients deny the snapshot
            inventory = {
                "project_tree_rss_bytes": 0,
                "project_tree_rss_upper_bound_bytes": 0,
                "project_processes": [],
                "processes_seen": 0,
                "cmdline_reads": 0,
                "cmdline_cache_hits": 0,
                "exited_identities": 0,
                "reused_pids": [],
                "inspection_failures": [
                    {"pid": self.pid, "error": type(error).__name__, "rss_upper_bound": -1}
                ],
                "process_coverage_bounded": False,
            }
            scan_error = f"{type(error).__name__}: {error}"
        scan_finished = self.monotonic_ns()
        self.sequence += 1
        payload: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "session_sha256": self.session["descriptor_sha256"],
            "repository": str(self.repository),
            "run_root": str(self.run_root),
            "publisher_pid": self.pid,
            "publisher_create_time": self.create_time,
            "sequence": self.sequence,
            "scan_started_monotonic_ns": scan_started,
            "scan_finished_monotonic_ns": scan_finished,
            "published_monotonic_ns": self.monotonic_ns(),
            "published_wall_time_ns": self.wall_time_ns(),
            "publisher_scan_error": scan_error,
            **inventory,
        }
        signature = _signature(str(self.session["session_token"]), payload)
        snapshot = {**payload, "signature_sha256": signature}
        _atomic_write_json(self.snapshot_path, snapshot)
        return snapshot

    def serve(self, *, interval_seconds: float = DEFAULT_INTERVAL_SECONDS) -> None:
        """Publish in the foreground until interrupted by the launcher."""
        if not 1.0 <= interval_seconds <= DEFAULT_TTL_SECONDS / 2:
            raise ValueError("Publisher interval must be in [1, TTL/2] seconds")
        self.acquire()
        try:
            while True:
                begun = self.monotonic_ns()
                self.publish_once()
                elapsed = (self.monotonic_ns() - begun) / 1_000_000_000
                time.sleep(max(0.0, interval_seconds - elapsed))
        finally:
            self.close()


class SharedResourceMonitor:
    """ResourceMonitor-compatible client backed by a shared full inventory."""

    def __init__(
        self,
        repository: Path,
        *,
        session_path: Path,
        snapshot_path: Path,
        run_root: Path | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        psutil_module: object = psutil,
        commit_reader: Callable[[], int | None] = _windows_commit_headroom,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        platform_name: str = os.name,
    ) -> None:
        if not 1.0 <= ttl_seconds <= 60.0:
            raise ValueError("Snapshot TTL must be in [1,60] seconds")
        self.repository = repository.resolve()
        if run_root is None:
            session_hint = _read_json(session_path)
            hinted_run_root = str(session_hint.get("run_root", ""))
            if not hinted_run_root:
                raise ValueError("Shared-resource session has no run-root binding")
            run_root = Path(hinted_run_root)
        self.run_root = run_root.resolve()
        self.repository_text = str(self.repository)
        self.run_root_text = str(self.run_root)
        self.run_root_key = os.path.normcase(os.path.abspath(self.run_root))
        self.disk_anchor = self.run_root.anchor
        self.session = load_shared_resource_session(
            session_path,
            repository=self.repository,
            run_root=self.run_root,
        )
        self.snapshot_path = snapshot_path
        self.ttl_ns = int(ttl_seconds * 1_000_000_000)
        self.module = cast(_Psutil, psutil_module)
        self.commit_reader = commit_reader
        self.monotonic_ns = monotonic_ns
        self.platform_name = platform_name
        self._publisher_process: _PublisherProcess | None = None
        self._publisher_identity: tuple[int, float] | None = None
        self._calls = 0
        self._snapshot_reads = 0
        self._guard_seconds_total = 0.0
        self._max_guard_seconds = 0.0
        self._last_sequence = 0
        self._publisher_full_scans_seen = 0

    def _publisher_is_live(self, pid: int, create_time: float) -> bool:
        identity = (pid, create_time)
        try:
            if self._publisher_process is None or self._publisher_identity != identity:
                self._publisher_process = self.module.Process(pid)
                self._publisher_identity = identity
            process = self._publisher_process
            not_zombie = (
                True
                if self.platform_name == "nt"
                else process.status() != self.module.STATUS_ZOMBIE
            )
            return bool(
                process.pid == pid
                and process.is_running()
                and not_zombie
                and float(process.create_time()) == create_time
            )
        except (
            self.module.NoSuchProcess,
            self.module.ZombieProcess,
            self.module.AccessDenied,
            OSError,
        ):
            return False

    def _validate_snapshot(self, now_ns: int) -> tuple[bool, dict[str, Any]]:
        try:
            snapshot = _read_json(self.snapshot_path)
            self._snapshot_reads += 1
            signature = str(snapshot.pop("signature_sha256", ""))
            expected = _signature(str(self.session["session_token"]), snapshot)
            if not hmac.compare_digest(signature, expected):
                raise ValueError("snapshot signature mismatch")
            if snapshot.get("schema") != SNAPSHOT_SCHEMA:
                raise ValueError("snapshot schema mismatch")
            if snapshot.get("session_sha256") != self.session["descriptor_sha256"]:
                raise ValueError("snapshot session mismatch")
            if snapshot.get("repository") != self.repository_text:
                raise ValueError("snapshot repository mismatch")
            if snapshot.get("run_root") != self.run_root_text:
                raise ValueError("snapshot run-root mismatch")
            started = _strict_int(
                snapshot.get("scan_started_monotonic_ns"),
                "scan start",
            )
            finished = _strict_int(
                snapshot.get("scan_finished_monotonic_ns"),
                "scan finish",
            )
            published = _strict_int(
                snapshot.get("published_monotonic_ns"),
                "publish time",
            )
            _strict_int(snapshot.get("published_wall_time_ns"), "publish wall time")
            if not started <= finished <= published <= now_ns:
                raise ValueError("snapshot monotonic timestamps are invalid or future-dated")
            age_ns = now_ns - started
            if age_ns > self.ttl_ns:
                raise ValueError("snapshot is stale")
            sequence = _strict_int(snapshot.get("sequence"), "sequence", minimum=1)
            if sequence < self._last_sequence:
                raise ValueError("snapshot sequence regressed")
            pid = _strict_int(snapshot.get("publisher_pid"), "publisher pid", minimum=1)
            created = _strict_finite_number(
                snapshot.get("publisher_create_time"),
                "publisher create_time",
                positive=True,
            )
            _validate_inventory_payload(snapshot)
            if not self._publisher_is_live(pid, created):
                raise ValueError("publisher PID identity is not live")
            self._last_sequence = sequence
            self._publisher_full_scans_seen = max(self._publisher_full_scans_seen, sequence)
            return True, {**snapshot, "signature_sha256": signature, "snapshot_age_ns": age_ns}
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return False, {
                "snapshot_valid": False,
                "snapshot_error": f"{type(error).__name__}: {error}",
            }

    def guard(
        self,
        run_root: Path,
        *,
        checkpoint_reservation: int,
        completed_updates: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Run cheap local checks and verify one fresh shared inventory."""
        begun = time.perf_counter()
        available = int(self.module.virtual_memory().available)
        disk = int(self.module.disk_usage(self.disk_anchor).free)
        commit = self.commit_reader()
        snapshot_valid, snapshot = self._validate_snapshot(self.monotonic_ns())
        run_root_binding_valid = os.path.normcase(os.path.abspath(run_root)) == self.run_root_key
        values: dict[str, Any] = {
            "host_available_bytes": available,
            "windows_commit_headroom_bytes": commit,
            "disk_free_bytes": disk,
            "checkpoint_reservation_bytes": checkpoint_reservation,
            "disk_free_after_reservation_bytes": disk - checkpoint_reservation,
            "full_process_scan": False,
            "completed_updates": completed_updates,
            "snapshot_valid": snapshot_valid,
            "run_root_binding_valid": run_root_binding_valid,
            **snapshot,
        }
        bounded = bool(snapshot.get("process_coverage_bounded", False))
        upper = int(snapshot.get("project_tree_rss_upper_bound_bytes", PROJECT_RSS_LIMIT_BYTES + 1))
        allowed = bool(
            snapshot_valid
            and run_root_binding_valid
            and bounded
            and snapshot.get("publisher_scan_error") is None
            and upper <= PROJECT_RSS_LIMIT_BYTES
            and available >= 2 * GIB
            and (commit is None or commit >= 2 * GIB)
            and disk - checkpoint_reservation >= 10_000_000_000
        )
        elapsed = time.perf_counter() - begun
        self._calls += 1
        self._guard_seconds_total += elapsed
        self._max_guard_seconds = max(self._max_guard_seconds, elapsed)
        values["guard_seconds"] = elapsed
        values["monitor_statistics"] = self.statistics()
        return allowed, values

    def statistics(self) -> dict[str, int | float]:
        """Expose compatible counters and distinguish publisher/client work."""
        return {
            "calls": self._calls,
            "full_scans": 0,
            "guard_seconds_total": self._guard_seconds_total,
            "full_guard_seconds_total": 0.0,
            "max_guard_seconds": self._max_guard_seconds,
            "client_snapshot_reads": self._snapshot_reads,
            "publisher_full_scans_seen": self._publisher_full_scans_seen,
        }


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Publish shared final-refit resource inventories")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize", help="create an immutable session")
    publish = subparsers.add_parser("publish", help="run the foreground publisher")
    clear = subparsers.add_parser(
        "clear-stale-lock",
        help="remove a lock only after proving its publisher identity is dead",
    )
    for command in (initialize, publish, clear):
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--session", type=Path, required=True)
    publish.add_argument("--snapshot", type=Path, required=True)
    publish.add_argument("--lock", type=Path, required=True)
    publish.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    clear.add_argument("--lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; the publisher remains owned by the invoking launcher."""
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "initialize":
        create_shared_resource_session(
            arguments.session,
            repository=arguments.repository,
            run_root=arguments.run_root,
        )
        return 0
    if arguments.command == "clear-stale-lock":
        clear_stale_publisher_lock(
            arguments.lock,
            session_path=arguments.session,
            repository=arguments.repository,
            run_root=arguments.run_root,
        )
        return 0
    publisher = SharedResourcePublisher(
        arguments.repository,
        arguments.run_root,
        session_path=arguments.session,
        snapshot_path=arguments.snapshot,
        lock_path=arguments.lock,
    )
    try:
        publisher.serve(interval_seconds=arguments.interval_seconds)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
