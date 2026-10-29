"""Turn H8 RAM boundaries into bounded backpressure without skipping work."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from operational.train40_system.durable_io import atomic_json
from operational.train40_system.resource_monitor import HOST_AVAILABLE_MIN_BYTES
from operational.train40_system.resource_monitor_receiver import ContinuousC2FResourceMonitor

TREE_FLOW_LIMIT_BYTES = 23_000_000_000
HOST_RECOVERY_BYTES = 3 * 1024**3
RSS_THROTTLE_SECONDS = 0.5
HOST_WAIT_SECONDS = 1.0
WAIT_PUBLICATION_SECONDS = 10.0


def _host_available_bytes() -> int:
    """Read current physical headroom without waiting for the five-second inventory."""
    return int(psutil.virtual_memory().available)


class MemoryFlowMonitor(ContinuousC2FResourceMonitor):
    """Wait for host memory and throttle aggregate RSS while retaining every other guard."""

    def __init__(
        self,
        output: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        host_available: Callable[[], int] = _host_available_bytes,
    ) -> None:
        super().__init__(output)
        self._flow_clock = clock
        self._flow_sleep = sleep
        self._flow_host_available = host_available
        self._flow_state = "NORMAL"
        self._flow_wait_seconds = 0.0
        self._flow_throttle_seconds = 0.0
        self._flow_wait_iterations = 0
        self._flow_throttle_count = 0
        self._flow_publications = 0
        self._flow_last_publication = float("-inf")

    def _publish_flow(
        self,
        state: str,
        telemetry: dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        now = self._flow_clock()
        transition = state != self._flow_state
        periodic_wait = (
            state == "WAITING_FOR_HOST_MEMORY"
            and now - self._flow_last_publication >= WAIT_PUBLICATION_SECONDS
        )
        initial = self._flow_publications == 0
        self._flow_state = state
        if not (force or transition or periodic_wait or initial):
            return
        atomic_json(
            self.output / "H8_MEMORY_FLOW.json",
            {
                "status": state,
                "tree_rss_bytes": telemetry.get("tree_rss_bytes"),
                "host_available_bytes": telemetry.get("host_available_bytes"),
                "monitor_sequence": telemetry.get("monitor_sequence"),
                "reasons": list(telemetry.get("reasons", [])),
                "wait_seconds": self._flow_wait_seconds,
                "throttle_seconds": self._flow_throttle_seconds,
                "wait_iterations": self._flow_wait_iterations,
                "throttle_count": self._flow_throttle_count,
                "tree_flow_limit_bytes": TREE_FLOW_LIMIT_BYTES,
                "host_available_min_bytes": HOST_AVAILABLE_MIN_BYTES,
                "host_recovery_bytes": HOST_RECOVERY_BYTES,
                "optimizer_updates": 0,
                "scientific_negative": False,
                "checked_utc": datetime.now(UTC).isoformat(),
            },
        )
        self._flow_publications += 1
        self._flow_last_publication = now

    @staticmethod
    def _non_ram_reasons(telemetry: dict[str, Any]) -> list[str]:
        return [
            str(reason)
            for reason in telemetry.get("reasons", [])
            if reason != "RAM_RESOURCE_BOUNDARY"
        ]

    def _throttle(self, telemetry: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        begun = self._flow_clock()
        self._flow_sleep(RSS_THROTTLE_SECONDS)
        elapsed = max(0.0, self._flow_clock() - begun)
        self._flow_throttle_seconds += elapsed
        self._flow_throttle_count += 1
        result = {
            **telemetry,
            "reasons": self._non_ram_reasons(telemetry),
            "memory_pressure": True,
            "memory_flow_state": "RSS_THROTTLED",
            "memory_flow_throttle_seconds": elapsed,
        }
        self._publish_flow("RSS_THROTTLED", result)
        return True, result

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Retain all non-RAM failures; turn RAM-only failures into waiting or throttling."""
        allowed, telemetry = super().guard(output)
        non_ram = self._non_ram_reasons(telemetry)
        if non_ram:
            self._publish_flow("BLOCKED_OTHER_RESOURCE", telemetry)
            return False, telemetry

        try:
            fresh_host = self._flow_host_available()
            if isinstance(fresh_host, bool) or not isinstance(fresh_host, int) or fresh_host < 0:
                raise ValueError("host available bytes must be a nonnegative integer")
        except BaseException as error:
            result = {
                **telemetry,
                "reasons": [
                    *non_ram,
                    f"MEMORY_FLOW_HOST_PROBE_FAILED:{type(error).__name__}",
                ],
                "memory_pressure": True,
                "memory_flow_state": "BLOCKED_HOST_PROBE",
            }
            self._publish_flow("BLOCKED_HOST_PROBE", result)
            return False, result
        telemetry = {
            **telemetry,
            "monitor_host_available_bytes": telemetry.get("host_available_bytes"),
            "host_available_bytes": fresh_host,
            "host_available_source": "fresh_psutil",
        }
        host_value = fresh_host
        tree_value = telemetry.get("tree_rss_bytes")
        if not isinstance(host_value, int) or not isinstance(tree_value, int):
            result = {
                **telemetry,
                "reasons": [*telemetry.get("reasons", []), "MEMORY_FLOW_TELEMETRY_MISSING"],
                "memory_pressure": True,
                "memory_flow_state": "BLOCKED_MISSING_TELEMETRY",
            }
            self._publish_flow("BLOCKED_MISSING_TELEMETRY", result)
            return False, result

        if host_value < HOST_AVAILABLE_MIN_BYTES:
            self._publish_flow("WAITING_FOR_HOST_MEMORY", telemetry, force=True)
            while True:
                begun = self._flow_clock()
                self._flow_sleep(HOST_WAIT_SECONDS)
                self._flow_wait_seconds += max(0.0, self._flow_clock() - begun)
                self._flow_wait_iterations += 1
                allowed, telemetry = super().guard(output)
                non_ram = self._non_ram_reasons(telemetry)
                if non_ram:
                    self._publish_flow("BLOCKED_OTHER_RESOURCE", telemetry, force=True)
                    return False, telemetry
                try:
                    fresh_host = self._flow_host_available()
                    if (
                        isinstance(fresh_host, bool)
                        or not isinstance(fresh_host, int)
                        or fresh_host < 0
                    ):
                        raise ValueError("host available bytes must be a nonnegative integer")
                except BaseException as error:
                    result = {
                        **telemetry,
                        "reasons": [
                            *non_ram,
                            f"MEMORY_FLOW_HOST_PROBE_FAILED:{type(error).__name__}",
                        ],
                        "memory_pressure": True,
                        "memory_flow_state": "BLOCKED_HOST_PROBE",
                    }
                    self._publish_flow("BLOCKED_HOST_PROBE", result, force=True)
                    return False, result
                telemetry = {
                    **telemetry,
                    "monitor_host_available_bytes": telemetry.get("host_available_bytes"),
                    "host_available_bytes": fresh_host,
                    "host_available_source": "fresh_psutil",
                }
                host_value = fresh_host
                tree_value = telemetry.get("tree_rss_bytes")
                if not isinstance(host_value, int) or not isinstance(tree_value, int):
                    result = {
                        **telemetry,
                        "reasons": [
                            *telemetry.get("reasons", []),
                            "MEMORY_FLOW_TELEMETRY_MISSING",
                        ],
                        "memory_pressure": True,
                        "memory_flow_state": "BLOCKED_MISSING_TELEMETRY",
                    }
                    self._publish_flow("BLOCKED_MISSING_TELEMETRY", result, force=True)
                    return False, result
                if host_value >= HOST_RECOVERY_BYTES:
                    break
                self._publish_flow("WAITING_FOR_HOST_MEMORY", telemetry)

        if tree_value >= TREE_FLOW_LIMIT_BYTES:
            return self._throttle(telemetry)
        result = {
            **telemetry,
            "reasons": [],
            "memory_pressure": False,
            "memory_flow_state": "NORMAL",
        }
        self._publish_flow("NORMAL", result)
        return bool(allowed or not result["reasons"]), result

    def snapshot(self) -> dict[str, Any]:
        """Add cumulative flow-control time without changing monitor telemetry."""
        state = super().snapshot()
        state["memory_flow"] = {
            "state": self._flow_state,
            "wait_seconds": self._flow_wait_seconds,
            "throttle_seconds": self._flow_throttle_seconds,
            "wait_iterations": self._flow_wait_iterations,
            "throttle_count": self._flow_throttle_count,
            "publications": self._flow_publications,
            "tree_flow_limit_bytes": TREE_FLOW_LIMIT_BYTES,
            "host_available_min_bytes": HOST_AVAILABLE_MIN_BYTES,
            "host_recovery_bytes": HOST_RECOVERY_BYTES,
        }
        return state


__all__ = ["MemoryFlowMonitor"]
