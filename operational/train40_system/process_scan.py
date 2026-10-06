"""Fast, privacy-preserving heavy-process scan for the frozen TRAIN40 guard."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import psutil

from operational.train40_system import models

_ORIGINAL_RESOURCE_GUARD = models.resource_guard
_SCAN_INTERVAL_SECONDS = 5.0
_LEGACY_MARKERS = (
    "operational.train40_system.models",
    "operational.train40_system.teacher",
    "operational.efficient_context.garl_train",
    "operational.efficient_context.garl_heads",
    "operational.efficient_context.run train",
)
_ENGINE_MODULE = "operational.train40_system.engine"
_GRAPH_ADMISSION_MODULE = "operational.train40_system.graph_launch_admission"
_LOCK = Lock()
_STATS: dict[str, int | float] = {
    "guard_calls": 0,
    "total_guard_seconds": 0.0,
    "total_delegate_seconds": 0.0,
    "scan_count": 0,
    "total_scan_seconds": 0.0,
    "last_scan_seconds": 0.0,
    "processes_seen": 0,
    "python_cmdline_reads": 0,
    "skipped_non_python": 0,
    "cmdline_read_failures": 0,
}


class _Candidate(Protocol):
    pid: int
    info: dict[str, str | None]

    def cmdline(self) -> list[str]: ...


class _CurrentProcess(Protocol):
    pid: int

    def parents(self) -> list[_Candidate]: ...

    def children(self, *, recursive: bool) -> list[_Candidate]: ...


class _PsutilModule(Protocol):
    def Process(self) -> _CurrentProcess: ...  # noqa: N802

    def process_iter(self, attrs: list[str]) -> Iterable[_Candidate]: ...


def _is_heavy_command(parts: list[str] | tuple[str, ...]) -> bool:
    """Apply every legacy marker and the admitted engine/probe module families."""
    lowered = tuple(str(part).lower() for part in parts)
    line = " ".join(lowered)
    if any(marker in line for marker in _LEGACY_MARKERS):
        return True
    if "train" in line and any(f"stage{stage}" in line for stage in range(70, 77)):
        return True
    for index, part in enumerate(lowered[1:], start=1):
        if lowered[index - 1] != "-m":
            continue
        if part == _ENGINE_MODULE or part.startswith(f"{_ENGINE_MODULE}_"):
            return True
        if part == _GRAPH_ADMISSION_MODULE:
            return True
    return False


def _scan(
    psutil_module: object = psutil,
    ignored_errors: tuple[type[BaseException], ...] = (
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
    ),
) -> tuple[list[int], dict[str, int]]:
    """Read command lines only for Python processes and return unrelated heavy PIDs."""
    module = cast("_PsutilModule", psutil_module)
    process = module.Process()
    related = {
        process.pid,
        *(candidate.pid for candidate in process.parents()),
        *(candidate.pid for candidate in process.children(recursive=True)),
    }
    others: list[int] = []
    counts = {
        "processes_seen": 0,
        "python_cmdline_reads": 0,
        "skipped_non_python": 0,
        "cmdline_read_failures": 0,
    }
    for candidate in module.process_iter(["name"]):
        counts["processes_seen"] += 1
        if candidate.pid in related:
            continue
        if not (candidate.info.get("name") or "").lower().startswith("python"):
            counts["skipped_non_python"] += 1
            continue
        counts["python_cmdline_reads"] += 1
        try:
            command = candidate.cmdline() or []
        except ignored_errors:
            counts["cmdline_read_failures"] += 1
            command = []
        if _is_heavy_command(command):
            others.append(candidate.pid)
    return others, counts


def resource_guard(output: Path) -> tuple[bool, dict]:
    """Replace only the heavy scan, then call the captured frozen guard exactly once."""
    guard_started = time.perf_counter()
    with _LOCK:
        now = time.monotonic()
        if now - models._heavy_scan_at >= _SCAN_INTERVAL_SECONDS:
            scan_started = time.perf_counter()
            others, counts = _scan()
            scan_seconds = time.perf_counter() - scan_started
            models._other_heavy_pids = others
            models._heavy_scan_at = time.monotonic()
            _STATS["scan_count"] += 1
            _STATS["total_scan_seconds"] += scan_seconds
            _STATS["last_scan_seconds"] = scan_seconds
            for name, value in counts.items():
                _STATS[name] += value

        # Suppress the captured guard's old scan even if a cadence boundary is
        # crossed between the wrapper's check and delegation. Restore the real
        # cadence stamp so frequent calls cannot postpone the next fast scan.
        cadence_stamp = models._heavy_scan_at
        models._heavy_scan_at = time.monotonic()
        delegate_started = time.perf_counter()
        try:
            result = _ORIGINAL_RESOURCE_GUARD(output)
        finally:
            delegate_seconds = time.perf_counter() - delegate_started
            models._heavy_scan_at = cadence_stamp
            _STATS["guard_calls"] += 1
            _STATS["total_delegate_seconds"] += delegate_seconds
            _STATS["total_guard_seconds"] += time.perf_counter() - guard_started
        return result


def snapshot() -> dict[str, int | float | list[int] | str]:
    """Return JSON-safe aggregate timing without process command lines."""
    with _LOCK:
        return {
            "schema": "train40_fast_process_scan_runtime_v1",
            **_STATS,
            "cached_other_heavy_pids": list(models._other_heavy_pids),
        }


__all__ = ["resource_guard", "snapshot"]
