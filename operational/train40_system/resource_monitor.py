"""Independent CPU resource monitor for one TRAIN40 trainer process."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

import psutil

SCAN_INTERVAL_SECONDS = 5.0
STALE_AFTER_SECONDS = 12.0
START_TIMEOUT_SECONDS = 30.0
TREE_RSS_LIMIT_BYTES = 16_000_000_000
HOST_AVAILABLE_MIN_BYTES = 2 * 1024**3
DISK_FREE_MIN_BYTES = 20_000_000_000
PHYSICAL_UPDATE_LIMIT = 240_000
RECOVERY_LIMIT = 2_000

_LEGACY_MARKERS = (
    "operational.train40_system.models",
    "operational.train40_system.teacher",
    "operational.efficient_context.garl_train",
    "operational.efficient_context.garl_heads",
    "operational.efficient_context.run train",
)
_ENGINE_MODULE = "operational.train40_system.engine"
_GRAPH_ADMISSION_MODULE = "operational.train40_system.graph_launch_admission"


class _Connection(Protocol):
    def poll(self, timeout: float = 0.0) -> bool: ...

    def recv(self) -> object: ...

    def close(self) -> None: ...


class _IdentityProcess(Protocol):
    pid: int

    def create_time(self) -> float: ...


class _Sender(Protocol):
    def send(self, value: object) -> None: ...

    def close(self) -> None: ...


class _StopEvent(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float) -> bool: ...


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _is_heavy_command(parts: list[str] | tuple[str, ...]) -> bool:
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


def _identity(process: _IdentityProcess) -> tuple[int, float]:
    return int(process.pid), float(process.create_time())


def _collect_sample(
    root_pid: int,
    root_create_time: float,
    output: Path,
    *,
    psutil_module: object = psutil,
    disk_usage: Callable[[Path], object] = shutil.disk_usage,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Collect one bounded resource snapshot without importing trainer modules."""
    module = cast(Any, psutil_module)
    root = module.Process(root_pid)
    if _identity(root) != (root_pid, root_create_time):
        raise RuntimeError("Trainer PID identity changed")
    parents = list(root.parents())
    children = list(root.children(recursive=True))
    related = {(root_pid, root_create_time)}
    related.update(_identity(process) for process in parents)
    related.update(_identity(process) for process in children)

    rss = int(root.memory_info().rss)
    live_children = 0
    for child in children:
        try:
            expected_identity = _identity(child)
            fresh = module.Process(child.pid)
            if _identity(fresh) != expected_identity:
                continue
            rss += int(fresh.memory_info().rss)
            live_children += 1
        except (module.NoSuchProcess, module.ZombieProcess):
            continue

    others: list[dict[str, int | float]] = []
    unreadable: list[int] = []
    processes_seen = 0
    python_cmdline_reads = 0
    for candidate in module.process_iter(["name"]):
        processes_seen += 1
        name = (candidate.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        python_cmdline_reads += 1
        try:
            identity = _identity(candidate)
            if identity in related:
                continue
            command = candidate.cmdline() or []
        except (module.AccessDenied, module.NoSuchProcess, module.ZombieProcess):
            unreadable.append(int(candidate.pid))
            continue
        if _is_heavy_command(command):
            others.append({"pid": identity[0], "create_time": identity[1]})

    memory = module.virtual_memory()
    disk = cast(Any, disk_usage(output))
    return {
        "sampled_monotonic": float(clock()),
        "root_pid": root_pid,
        "root_create_time": root_create_time,
        "tree_rss_bytes": rss,
        "tree_children": live_children,
        "host_available_bytes": int(memory.available),
        "disk_free_bytes": int(disk.free),
        "other_heavy_processes": others,
        "inventory_unreadable_python_pids": unreadable,
        "processes_seen": processes_seen,
        "python_cmdline_reads": python_cmdline_reads,
    }


def _worker(
    root_pid: int,
    root_create_time: float,
    output: str,
    sender: _Sender,
    stop_event: _StopEvent,
) -> None:
    """Publish immediate and periodic small snapshots; never import Torch."""
    sequence = 0
    try:
        while not stop_event.is_set():
            begun = time.perf_counter()
            try:
                if "torch" in sys.modules:
                    raise RuntimeError("Torch imported in resource monitor process")
                sample = _collect_sample(root_pid, root_create_time, Path(output))
                sample.update(
                    {
                        "status": "OK",
                        "sequence": sequence,
                        "scan_seconds": time.perf_counter() - begun,
                        "torch_imported": "torch" in sys.modules,
                    }
                )
            except BaseException as error:
                sample = {
                    "status": "ERROR",
                    "sequence": sequence,
                    "sampled_monotonic": time.monotonic(),
                    "error": f"{type(error).__name__}: {error}",
                    "torch_imported": "torch" in sys.modules,
                }
            try:
                sender.send(sample)
            except (BrokenPipeError, EOFError, OSError):
                break
            sequence += 1
            elapsed = time.perf_counter() - begun
            stop_event.wait(max(0.0, SCAN_INTERVAL_SECONDS - elapsed))
    finally:
        sender.close()


def _budget_state(output: Path) -> dict[str, Any]:
    authorization = _read(output / "AUTHORIZATION.json")
    technical = _read(output / "TECHNICAL_ACCOUNTING.json")
    physical = int(authorization["previous_physical_execution_upper"])
    physical += int(technical["synthetic_optimizer_updates"])
    recovery = 0
    journals = 0
    pending = 0
    for path in (output / "fits").glob("*/UPDATE_JOURNAL.json"):
        journal = _read(path)
        journals += 1
        committed = int(journal["committed_updates"])
        recovered = int(journal["recovery_upper"])
        pending_here = int(journal["pending_update_upper"])
        physical += committed + recovered + pending_here
        recovery += recovered
        pending += pending_here
    pause_markers = [
        name
        for name in ("COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST")
        if (output / name).is_file()
    ]
    return {
        "physical_updates_upper": physical,
        "new_recovery_upper": recovery,
        "pending_update_upper": pending,
        "journal_count": journals,
        "deadline_utc": authorization["deadline_utc"],
        "pause_markers": pause_markers,
    }


class ResourceMonitor:
    """Own one spawn monitor and retain exact cheap per-update admission checks."""

    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self._context = multiprocessing.get_context("spawn")
        self._receiver: _Connection | None = None
        self._process: Any = None
        self._stop_event: Any = None
        self._latest: dict[str, Any] | None = None
        self._started = False
        self._closed = False
        self._guard_calls = 0
        self._guard_total_seconds = 0.0
        self._last_guard_seconds = 0.0
        self._snapshots_received = 0
        self._lock = RLock()

    def start(self) -> None:
        """Spawn the CPU monitor and require its first complete snapshot."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Resource monitor is closed")
            if self._started:
                return
            root = psutil.Process()
            receiver, sender = self._context.Pipe(duplex=False)
            stop_event = self._context.Event()
            process = self._context.Process(
                target=_worker,
                args=(root.pid, root.create_time(), str(self.output), sender, stop_event),
                name="train40-resource-monitor",
                daemon=True,
            )
            process.start()
            sender.close()
            self._receiver = cast(_Connection, receiver)
            self._process = process
            self._stop_event = stop_event
            deadline = time.monotonic() + START_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if self._receiver.poll(0.1):
                    value = self._receiver.recv()
                    if not isinstance(value, dict):
                        self.close()
                        raise RuntimeError("Resource monitor returned a non-object snapshot")
                    self._latest = value
                    self._snapshots_received = 1
                    self._started = True
                    if value.get("status") != "OK":
                        error = value.get("error", "unknown monitor error")
                        self.close()
                        raise RuntimeError(f"Initial resource monitor sample failed: {error}")
                    return
                if not process.is_alive():
                    self.close()
                    raise RuntimeError("Resource monitor exited before its first snapshot")
            self.close()
            raise TimeoutError("Resource monitor did not publish its first snapshot")

    def _drain(self) -> None:
        if self._receiver is None:
            return
        try:
            while self._receiver.poll():
                value = self._receiver.recv()
                if isinstance(value, dict):
                    self._latest = value
                    self._snapshots_received += 1
        except (EOFError, OSError):
            return

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Admit one update using fresh monitor state and exact cheap local checks."""
        begun = time.perf_counter()
        try:
            return self._guard_once(output)
        finally:
            elapsed = time.perf_counter() - begun
            with self._lock:
                self._guard_total_seconds += elapsed
                self._last_guard_seconds = elapsed

    def _guard_once(self, output: Path) -> tuple[bool, dict[str, Any]]:
        if output.resolve() != self.output:
            raise ValueError("Resource monitor output differs from its bound root")
        with self._lock:
            self._guard_calls += 1
            self._drain()
            reasons: list[str] = []
            try:
                budget = _budget_state(self.output)
            except BaseException as error:
                budget = {
                    "physical_updates_upper": -1,
                    "new_recovery_upper": -1,
                    "pending_update_upper": -1,
                    "journal_count": -1,
                    "deadline_utc": None,
                    "pause_markers": [],
                    "error": f"{type(error).__name__}: {error}",
                }
                reasons.append("BUDGET_CHECK_FAILED")
            if budget["physical_updates_upper"] >= PHYSICAL_UPDATE_LIMIT:
                reasons.append("PHYSICAL_UPDATE_CAP")
            if budget["new_recovery_upper"] >= RECOVERY_LIMIT:
                reasons.append("NEW_RECOVERY_RESERVE_EXHAUSTED")
            if budget["deadline_utc"] is not None and datetime.now(UTC) >= datetime.fromisoformat(
                str(budget["deadline_utc"])
            ):
                reasons.append("USER_DEADLINE")
            if budget["pause_markers"]:
                reasons.append("COORDINATION_PAUSE_REQUESTED")

            sample = self._latest
            process_alive = bool(self._process is not None and self._process.is_alive())
            age = None
            if not self._started:
                reasons.append("RESOURCE_MONITOR_NOT_STARTED")
            if not process_alive:
                reasons.append("RESOURCE_MONITOR_PROCESS_DEAD")
            if sample is None:
                reasons.append("RESOURCE_MONITOR_NO_SNAPSHOT")
            else:
                age = time.monotonic() - float(sample["sampled_monotonic"])
                if age > STALE_AFTER_SECONDS:
                    reasons.append("RESOURCE_MONITOR_STALE")
                if sample.get("status") != "OK":
                    reasons.append("RESOURCE_MONITOR_ERROR")
                if sample.get("inventory_unreadable_python_pids"):
                    reasons.append("RESOURCE_MONITOR_INVENTORY_INCOMPLETE")
                if sample.get("other_heavy_processes"):
                    reasons.append("ANOTHER_HEAVY_TRAINER")
                if int(sample.get("tree_rss_bytes", TREE_RSS_LIMIT_BYTES)) >= TREE_RSS_LIMIT_BYTES:
                    reasons.append("RAM_RESOURCE_BOUNDARY")
                if int(sample.get("host_available_bytes", 0)) < HOST_AVAILABLE_MIN_BYTES:
                    if "RAM_RESOURCE_BOUNDARY" not in reasons:
                        reasons.append("RAM_RESOURCE_BOUNDARY")
                if int(sample.get("disk_free_bytes", 0)) < DISK_FREE_MIN_BYTES:
                    reasons.append("CHECKPOINT_DISK_RESERVE")
            telemetry = {
                **budget,
                "tree_rss_bytes": None if sample is None else sample.get("tree_rss_bytes"),
                "host_available_bytes": (
                    None if sample is None else sample.get("host_available_bytes")
                ),
                "disk_free_bytes": None if sample is None else sample.get("disk_free_bytes"),
                "other_heavy_pids": (
                    []
                    if sample is None
                    else [entry["pid"] for entry in sample.get("other_heavy_processes", [])]
                ),
                "monitor_sequence": None if sample is None else sample.get("sequence"),
                "monitor_age_seconds": age,
                "monitor_process_alive": process_alive,
                "reasons": reasons,
            }
            return not reasons, telemetry

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe aggregate state without command lines."""
        with self._lock:
            self._drain()
            return {
                "schema": "train40_resource_monitor_runtime_v1",
                "started": self._started,
                "closed": self._closed,
                "guard_calls": self._guard_calls,
                "guard_total_seconds": self._guard_total_seconds,
                "last_guard_seconds": self._last_guard_seconds,
                "snapshots_received": self._snapshots_received,
                "monitor_pid": None if self._process is None else self._process.pid,
                "monitor_process_alive": bool(
                    self._process is not None and self._process.is_alive()
                ),
                "latest": self._latest,
            }

    def close(self) -> None:
        """Stop only the monitor process owned by this instance."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._stop_event is not None:
                self._stop_event.set()
            if self._process is not None:
                self._process.join(timeout=SCAN_INTERVAL_SECONDS + 2.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            if self._receiver is not None:
                self._receiver.close()


__all__ = ["ResourceMonitor"]
