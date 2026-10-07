"""Continuously drain the independent resource monitor while H8 prepares jobs."""

from __future__ import annotations

import math
import time
from pathlib import Path
from threading import Event, Thread, current_thread
from typing import Any

from operational.train40_system.engine_c2f_graph_replay import C2FResourceMonitor

RECEIVER_POLL_SECONDS = 0.5
RECEIVER_JOIN_SECONDS = 2.0


class ContinuousC2FResourceMonitor(C2FResourceMonitor):
    """Keep monitor snapshots fresh even when the trainer does not call ``guard``.

    The inherited lock serializes this receiver with ``guard`` and ``snapshot``.
    All admission decisions, budgets, limits, freshness checks, and transient
    inventory repair remain in :class:`C2FResourceMonitor`.
    """

    def __init__(self, output: Path) -> None:
        super().__init__(output)
        self._continuous_stop = Event()
        self._continuous_thread: Thread | None = None
        self._continuous_error: str | None = None

    def _record_receiver_error(self, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"
        if self._continuous_error is None:
            self._continuous_error = message
            previous = self._latest
            self._latest = {
                "status": "ERROR",
                "sequence": None if previous is None else previous.get("sequence"),
                "sampled_monotonic": time.monotonic(),
                "error": f"Continuous receiver failed: {message}",
                "torch_imported": (
                    None if previous is None else previous.get("torch_imported")
                ),
            }
        self._continuous_stop.set()

    @staticmethod
    def _integer(value: object, field: str, *, positive: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Resource monitor {field} must be an integer")
        if value < (1 if positive else 0):
            raise ValueError(f"Resource monitor {field} is out of range")
        return value

    @staticmethod
    def _finite(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Resource monitor {field} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Resource monitor {field} must be finite")
        return result

    def _validate_snapshot(self, value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise RuntimeError("Resource monitor returned a non-object snapshot")
        status = value.get("status")
        if status not in {"OK", "ERROR"}:
            raise ValueError("Resource monitor status must be OK or ERROR")
        self._integer(value.get("sequence"), "sequence")
        sampled = self._finite(value.get("sampled_monotonic"), "sampled_monotonic")
        if sampled > time.monotonic():
            raise ValueError("Resource monitor sampled_monotonic is in the future")
        if not isinstance(value.get("torch_imported"), bool):
            raise ValueError("Resource monitor torch_imported must be boolean")
        if status == "ERROR":
            if not isinstance(value.get("error"), str) or not value["error"]:
                raise ValueError("Resource monitor ERROR snapshot requires an error")
            return value

        if value["torch_imported"] is not False:
            raise ValueError("Torch imported in resource monitor process")
        root_pid = self._integer(value.get("root_pid"), "root_pid", positive=True)
        root_created = self._finite(value.get("root_create_time"), "root_create_time")
        previous = self._latest
        if previous is not None and previous.get("status") == "OK":
            if root_pid != previous.get("root_pid") or root_created != previous.get(
                "root_create_time"
            ):
                raise ValueError("Resource monitor root identity changed")
        for field in ("tree_rss_bytes", "host_available_bytes", "disk_free_bytes"):
            self._integer(value.get(field), field)
        heavy = value.get("other_heavy_processes")
        unreadable = value.get("inventory_unreadable_python_pids")
        if not isinstance(heavy, list) or not isinstance(unreadable, list):
            raise ValueError("Resource monitor inventory fields must be lists")
        for entry in heavy:
            if not isinstance(entry, dict):
                raise ValueError("Resource monitor heavy identity must be an object")
            self._integer(entry.get("pid"), "heavy pid", positive=True)
            self._finite(entry.get("create_time"), "heavy create_time")
        for pid in unreadable:
            self._integer(pid, "unreadable pid", positive=True)
        return value

    def _drain(self) -> None:
        """Drain under the inherited lock and retain receiver failures."""
        if self._receiver is None or self._continuous_error is not None:
            return
        try:
            while self._receiver.poll():
                value = self._validate_snapshot(self._receiver.recv())
                self._latest = value
                self._snapshots_received += 1
        except BaseException as error:
            self._record_receiver_error(error)

    def _receive_forever(self) -> None:
        while not self._continuous_stop.is_set():
            with self._lock:
                self._drain()
            self._continuous_stop.wait(RECEIVER_POLL_SECONDS)

    def start(self) -> None:
        """Start the original monitor, then continuously consume its snapshots."""
        super().start()
        with self._lock:
            try:
                self._latest = self._validate_snapshot(self._latest)
            except BaseException as error:
                self._record_receiver_error(error)
                self.close()
                raise RuntimeError("Initial resource monitor snapshot is invalid") from error
            if self._continuous_thread is not None and self._continuous_thread.is_alive():
                return
            if self._closed:
                raise RuntimeError("Resource monitor is closed")
            self._continuous_stop.clear()
            thread = Thread(
                target=self._receive_forever,
                name="train40-resource-monitor-receiver",
                daemon=True,
            )
            self._continuous_thread = thread
            thread.start()

    def snapshot(self) -> dict[str, Any]:
        """Add bounded receiver lifecycle telemetry to the original snapshot."""
        state = super().snapshot()
        with self._lock:
            state["continuous_receiver"] = {
                "poll_seconds": RECEIVER_POLL_SECONDS,
                "thread_alive": bool(
                    self._continuous_thread is not None
                    and self._continuous_thread.is_alive()
                ),
                "error": self._continuous_error,
            }
        return state

    def close(self) -> None:
        """Stop the receiver first, then close the process and pipe it consumes."""
        self._continuous_stop.set()
        thread = self._continuous_thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=RECEIVER_JOIN_SECONDS)
        super().close()


__all__ = ["ContinuousC2FResourceMonitor"]
