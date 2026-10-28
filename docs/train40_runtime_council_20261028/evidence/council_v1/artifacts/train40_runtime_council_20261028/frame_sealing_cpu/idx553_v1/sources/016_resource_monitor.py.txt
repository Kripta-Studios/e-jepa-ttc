"""Cached, fail-closed resource guards for the final TRAIN40 refit.

The cheap guard still reads host RAM, Windows commit headroom and disk on every
call.  Full process coverage is refreshed on the first call and every 25 updates.
Only immutable command lines are cached, keyed by ``(pid, create_time)``; RSS is
always read fresh during a full scan.
"""

from __future__ import annotations

import ctypes
import json
import time
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Protocol, cast

import psutil

GIB = 1024**3
FULL_SCAN_UPDATE_INTERVAL = 25


class _MemoryInfo(Protocol):
    rss: int


class _VirtualMemory(Protocol):
    available: int


class _DiskUsage(Protocol):
    free: int


class _Process(Protocol):
    pid: int

    def cmdline(self) -> list[str]: ...

    def create_time(self) -> float: ...

    def memory_info(self) -> _MemoryInfo: ...

    def oneshot(self) -> AbstractContextManager[None]: ...


class _Psutil(Protocol):
    AccessDenied: type[BaseException]
    NoSuchProcess: type[BaseException]
    ZombieProcess: type[BaseException]

    def process_iter(self, attrs: Sequence[str]) -> Sequence[_Process]: ...

    def virtual_memory(self) -> _VirtualMemory: ...

    def disk_usage(self, path: str) -> _DiskUsage: ...


@dataclass(frozen=True)
class ProcessInventory:
    """One complete, JSON-safe project-process inventory."""

    project_tree_rss_bytes: int
    project_tree_rss_upper_bound_bytes: int
    project_processes: tuple[dict[str, int | float], ...]
    processes_seen: int
    cmdline_reads: int
    cmdline_cache_hits: int
    exited_identities: int
    reused_pids: tuple[int, ...]
    inspection_failures: tuple[dict[str, int | str], ...]


def _windows_commit_headroom() -> int | None:
    if not hasattr(ctypes, "windll"):
        return None

    class Performance(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    counter = Performance()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counter), counter.cb):
        raise OSError("Windows commit counters unavailable")
    return int((counter.CommitLimit - counter.CommitTotal) * counter.PageSize)


class CachedProcessInventory:
    """Cache command lines without caching liveness, identity or memory usage."""

    def __init__(self, repository: Path, *, psutil_module: object = psutil) -> None:
        self.repository = repository.resolve()
        self.module = cast(_Psutil, psutil_module)
        self.markers = (
            str(self.repository).lower(),
            self.repository.name.lower(),
            "operational.rgb_port",
        )
        self._commands: dict[tuple[int, float], tuple[str, ...]] = {}
        self._unclassified: set[tuple[int, float]] = set()

    def scan(self) -> ProcessInventory:
        """Refresh every PID/create-time/RSS and fail closed on unreadable live entries."""
        prior_identities = set(self._commands) | self._unclassified
        seen: set[tuple[int, float]] = set()
        seen_pid_identity: dict[int, tuple[int, float]] = {}
        project: list[dict[str, int | float]] = []
        failures: list[dict[str, int | str]] = []
        reused: set[int] = set()
        reads = hits = processes_seen = 0
        ignored_gone = (self.module.NoSuchProcess, self.module.ZombieProcess)
        inspection_errors = ignored_gone + (self.module.AccessDenied, OSError)
        # psutil >=6 does not detect PID reuse in process_iter's internal cache.
        # Clearing it creates fresh Process identities while retaining our smaller,
        # explicit command cache keyed by PID and creation time.
        cache_clear = getattr(self.module.process_iter, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
        for discovered in self.module.process_iter(["pid"]):
            processes_seen += 1
            pid = int(discovered.pid)
            process = discovered
            identity: tuple[int, float] | None = None
            try:
                with process.oneshot():
                    created = float(process.create_time())
                    identity = (pid, created)
                    previous = seen_pid_identity.get(pid)
                    if previous is not None and previous != identity:
                        reused.add(pid)
                    seen_pid_identity[pid] = identity
                    if identity in self._unclassified:
                        rss_upper_bound = int(process.memory_info().rss)
                        failures.append(
                            {
                                "pid": pid,
                                "error": "CachedAccessDenied",
                                "rss_upper_bound": rss_upper_bound,
                            }
                        )
                        seen.add(identity)
                        continue
                    if identity in self._commands:
                        command_parts = self._commands[identity]
                        hits += 1
                    else:
                        command_parts = tuple(str(item) for item in (process.cmdline() or []))
                        self._commands[identity] = command_parts
                        reads += 1
                    command = " ".join(command_parts).lower()
                    matches = (
                        any(marker in command for marker in self.markers)
                        and "pytest" not in command
                    )
                    rss = int(process.memory_info().rss) if matches else 0
            except ignored_gone:
                continue
            except (self.module.AccessDenied, OSError) as error:
                # A protected process with an unreadable command might be part of the
                # project.  Bound it as project RSS when possible rather than silently
                # omitting it like the legacy guard.  If RSS is also unavailable the
                # failure is unbounded and the caller must deny the update.
                try:
                    rss_upper_bound = int(process.memory_info().rss)
                except inspection_errors:
                    rss_upper_bound = -1
                failures.append(
                    {
                        "pid": pid,
                        "error": type(error).__name__,
                        "rss_upper_bound": rss_upper_bound,
                    }
                )
                if identity is not None:
                    self._unclassified.add(identity)
                    seen.add(identity)
                continue
            if identity is None:  # pragma: no cover - defensive static narrowing
                raise AssertionError("process identity was not established")
            seen.add(identity)
            if matches:
                project.append({"pid": pid, "create_time": created, "rss": rss})

        old_identities = prior_identities - seen
        old_by_pid = {pid: identity for identity in old_identities for pid in (identity[0],)}
        for pid, identity in seen_pid_identity.items():
            if pid in old_by_pid and old_by_pid[pid] != identity:
                reused.add(pid)
        for identity in old_identities:
            self._commands.pop(identity, None)
        self._unclassified.intersection_update(seen)
        project.sort(key=lambda item: (int(item["pid"]), float(item["create_time"])))
        project_rss = sum(int(item["rss"]) for item in project)
        bounded_unknown_rss = sum(
            int(item["rss_upper_bound"]) for item in failures if int(item["rss_upper_bound"]) >= 0
        )
        return ProcessInventory(
            project_tree_rss_bytes=project_rss,
            project_tree_rss_upper_bound_bytes=project_rss + bounded_unknown_rss,
            project_processes=tuple(project),
            processes_seen=processes_seen,
            cmdline_reads=reads,
            cmdline_cache_hits=hits,
            exited_identities=len(old_identities),
            reused_pids=tuple(sorted(reused)),
            inspection_failures=tuple(failures),
        )


class ResourceMonitor:
    """Preserve cheap checks per update and full process coverage every 25 updates."""

    def __init__(
        self,
        repository: Path,
        *,
        psutil_module: object = psutil,
        commit_reader: Callable[[], int | None] = _windows_commit_headroom,
    ) -> None:
        self.module = cast(_Psutil, psutil_module)
        self.inventory = CachedProcessInventory(repository, psutil_module=psutil_module)
        self.commit_reader = commit_reader
        self.last_full_update: int | None = None
        self._calls = 0
        self._full_scans = 0
        self._guard_seconds_total = 0.0
        self._full_guard_seconds_total = 0.0
        self._max_guard_seconds = 0.0

    def guard(
        self,
        run_root: Path,
        *,
        checkpoint_reservation: int,
        completed_updates: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Run fresh cheap checks and a full scan at the unchanged 25-update cadence."""
        begun = time.perf_counter()
        available = int(self.module.virtual_memory().available)
        disk = int(self.module.disk_usage(str(run_root.resolve().anchor)).free)
        commit = self.commit_reader()
        full = (
            self.last_full_update is None
            or completed_updates < self.last_full_update
            or completed_updates - self.last_full_update >= FULL_SCAN_UPDATE_INTERVAL
        )
        values: dict[str, Any] = {
            "host_available_bytes": available,
            "windows_commit_headroom_bytes": commit,
            "disk_free_bytes": disk,
            "checkpoint_reservation_bytes": checkpoint_reservation,
            "disk_free_after_reservation_bytes": disk - checkpoint_reservation,
            "full_process_scan": full,
            "completed_updates": completed_updates,
        }
        allowed = (
            available >= 2 * GIB
            and (commit is None or commit >= 2 * GIB)
            and disk - checkpoint_reservation >= 10_000_000_000
        )
        if full:
            inventory = self.inventory.scan()
            self.last_full_update = completed_updates
            values.update(
                project_tree_rss_bytes=inventory.project_tree_rss_bytes,
                project_tree_rss_upper_bound_bytes=(inventory.project_tree_rss_upper_bound_bytes),
                project_processes=list(inventory.project_processes),
                processes_seen=inventory.processes_seen,
                cmdline_reads=inventory.cmdline_reads,
                cmdline_cache_hits=inventory.cmdline_cache_hits,
                exited_identities=inventory.exited_identities,
                reused_pids=list(inventory.reused_pids),
                inspection_failures=list(inventory.inspection_failures),
            )
            has_unbounded_failure = any(
                int(item["rss_upper_bound"]) < 0 for item in inventory.inspection_failures
            )
            values["process_coverage_bounded"] = not has_unbounded_failure
            allowed = bool(
                allowed
                and inventory.project_tree_rss_upper_bound_bytes <= 23_000_000_000
                and not has_unbounded_failure
            )
        elapsed = time.perf_counter() - begun
        self._calls += 1
        self._full_scans += int(full)
        self._guard_seconds_total += elapsed
        self._full_guard_seconds_total += elapsed if full else 0.0
        self._max_guard_seconds = max(self._max_guard_seconds, elapsed)
        values["guard_seconds"] = elapsed
        values["monitor_statistics"] = self.statistics()
        return allowed, values

    def statistics(self) -> dict[str, int | float]:
        """Return cumulative overhead counters suitable for the update journal."""
        return {
            "calls": self._calls,
            "full_scans": self._full_scans,
            "guard_seconds_total": self._guard_seconds_total,
            "full_guard_seconds_total": self._full_guard_seconds_total,
            "max_guard_seconds": self._max_guard_seconds,
        }


def recovery_upper(run_root: Path) -> int:
    """Count durable gap, pending boundary and prior recovery reserve exactly once."""
    from operational.rgb_port.accounting import read_json_shared

    total = 0
    for journal in (run_root / "fits").glob("*/UPDATE_JOURNAL.json"):
        value = read_json_shared(journal)
        completed = int(value.get("completed_updates", 0))
        durable = int(value.get("durable_updates", 0))
        total += (
            int(value.get("recovery_upper", 0))
            + max(0, completed - durable)
            + int(value.get("pending_update_upper", 0))
        )
    return total


def assert_recovery_budget(
    run_root: Path,
    *,
    prospective_pending: int,
    limit: int = 11_092,
) -> None:
    """Reject a boundary whose worst-case replay reserve exceeds the fixed cap."""
    if prospective_pending < 0:
        raise ValueError("prospective_pending must be non-negative")
    upper = recovery_upper(run_root) + prospective_pending
    if upper > limit:
        raise RuntimeError(f"RGB-PORT global recovery reserve exceeded: {upper} > {limit}")


def benchmark_against_legacy(
    run_root: Path,
    *,
    checkpoint_reservation: int,
    samples: int = 5,
) -> dict[str, Any]:
    """Compare stable-host full inventories and elapsed time without changing guards."""
    from operational.rgb_port.train_producers import _resource_guard

    if not 2 <= samples <= 20:
        raise ValueError("Resource benchmark samples must be in [2,20]")
    monitor = ResourceMonitor(Path(__file__).resolve().parents[2])
    legacy_rows: list[dict[str, Any]] = []
    cached_rows: list[dict[str, Any]] = []
    for index in range(samples):
        started = time.perf_counter()
        legacy_allowed, legacy = _resource_guard(
            run_root, checkpoint_reservation=checkpoint_reservation
        )
        legacy_seconds = time.perf_counter() - started
        cached_allowed, cached = monitor.guard(
            run_root,
            checkpoint_reservation=checkpoint_reservation,
            completed_updates=index * FULL_SCAN_UPDATE_INTERVAL,
        )
        legacy_rows.append({"allowed": legacy_allowed, "seconds": legacy_seconds, **legacy})
        cached_rows.append({"allowed": cached_allowed, **cached})
    cheap_rows = [
        monitor.guard(
            run_root,
            checkpoint_reservation=checkpoint_reservation,
            completed_updates=(samples - 1) * FULL_SCAN_UPDATE_INTERVAL + offset,
        )
        for offset in range(1, 11)
    ]
    legacy_populations = [
        sorted((int(item["pid"]), int(item["rss"])) for item in row["project_processes"])
        for row in legacy_rows
    ]
    cached_populations = [
        sorted((int(item["pid"]), int(item["rss"])) for item in row["project_processes"])
        for row in cached_rows
    ]
    pid_population_equal = [
        [pid for pid, _ in legacy] == [pid for pid, _ in cached]
        for legacy, cached in zip(legacy_populations, cached_populations, strict=True)
    ]
    rss_delta_bytes = [
        {
            str(pid): cached_rss - legacy_rss
            for (pid, legacy_rss), (_, cached_rss) in zip(legacy, cached, strict=True)
        }
        if pids_equal
        else {}
        for legacy, cached, pids_equal in zip(
            legacy_populations, cached_populations, pid_population_equal, strict=True
        )
    ]
    legacy_seconds = [float(item["seconds"]) for item in legacy_rows]
    cached_seconds = [float(item["guard_seconds"]) for item in cached_rows]
    legacy_median = median(legacy_seconds)
    cached_median = median(cached_seconds)
    return {
        "schema": "final_refit_resource_guard_benchmark_v1",
        "samples": samples,
        "pid_population_equal": pid_population_equal,
        "all_pid_populations_equal": all(pid_population_equal),
        "rss_delta_bytes_cached_minus_legacy": rss_delta_bytes,
        "legacy_seconds": legacy_seconds,
        "cached_seconds": cached_seconds,
        "legacy_median_seconds": legacy_median,
        "cached_median_seconds": cached_median,
        "median_speedup": legacy_median / cached_median if cached_median else None,
        "cheap_guard_seconds": [float(row[1]["guard_seconds"]) for row in cheap_rows],
        "cheap_guard_median_seconds": median(float(row[1]["guard_seconds"]) for row in cheap_rows),
        "all_cheap_guards_allowed": all(row[0] for row in cheap_rows),
        "legacy": legacy_rows,
        "cached": cached_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a bounded stable-host A/B report for review before integration."""
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-reservation", type=int, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = benchmark_against_legacy(
        args.run_root,
        checkpoint_reservation=args.checkpoint_reservation,
        samples=args.samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


__all__ = [
    "FULL_SCAN_UPDATE_INTERVAL",
    "CachedProcessInventory",
    "ProcessInventory",
    "ResourceMonitor",
    "assert_recovery_budget",
    "benchmark_against_legacy",
    "recovery_upper",
]


if __name__ == "__main__":
    raise SystemExit(main())
