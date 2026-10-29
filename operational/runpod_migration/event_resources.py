"""Actual cgroup and 600 GB event-volume guard without recursive volume scans."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.runpod_resources_v3 import memory_snapshot, parse_meminfo

GIB = 1024**3
EVENT_VOLUME_BYTES = 600_000_000_000
OUTPUT_LEDGER_RESERVE_BYTES = 50 * GIB
INFLIGHT_RESERVE_BYTES = 4 * GIB
VOLUME_SNAPSHOT_TTL_SECONDS = 120.0


class EventResourceMonitor:
    """Keep 2 GiB RAM and bounded event-volume reserves across all six workers."""

    deadline_timestamp: float | None = None

    def __init__(
        self,
        *args: object,
        budget_bytes: int = 61_000_000_000,
        snapshot_root: Path = Path("/workspace/ejepa_event_runs"),
        **kwargs: object,
    ) -> None:
        del args, kwargs
        if os.name != "posix":
            raise RuntimeError("EventResourceMonitor requires Linux")
        if budget_bytes < 8 * GIB:
            raise ValueError("Event cgroup budget is implausibly small")
        self.budget_bytes = budget_bytes
        self.snapshot_root = snapshot_root.resolve()

    def guard(
        self, run_root: Path, *, checkpoint_reservation: int, completed_updates: int
    ) -> tuple[bool, dict[str, Any]]:
        try:
            if self.deadline_timestamp is not None and time.time() >= self.deadline_timestamp:
                return False, {
                    "schema": "ejepa_event_resources_v1",
                    "reason": "CLOUD_EXECUTION_DEADLINE",
                    "pod_billing_stopped": False,
                    "host_available_bytes": 0,
                    "windows_commit_headroom_bytes": None,
                }
            if checkpoint_reservation < 0 or completed_updates < 0:
                raise ValueError("Negative resource reservation or update count")
            resolved_run = run_root.resolve()
            if resolved_run != self.snapshot_root and not resolved_run.is_relative_to(
                self.snapshot_root
            ):
                raise ValueError("Event run is outside the authorized snapshot root")
            mem = parse_meminfo(Path("/proc/meminfo").read_text())
            headroom, reports = memory_snapshot(
                Path("/sys/fs/cgroup"),
                Path("/proc/self/cgroup").read_text(),
                self.budget_bytes,
            )
            available = min(int(mem["MemAvailable"]), int(headroom))
            disk = int(shutil.disk_usage(run_root).free)
            snapshot_path = self.snapshot_root / "EVENT_VOLUME_SNAPSHOT.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            age = time.time() - float(snapshot["observed_timestamp"])
            if not (
                snapshot.get("schema") == "ejepa_event_volume_snapshot_v1"
                and snapshot.get("run_root") == str(self.snapshot_root)
                and int(snapshot.get("quota_bytes", -1)) == EVENT_VOLUME_BYTES
                and 0 <= int(snapshot.get("used_bytes", -1)) <= EVENT_VOLUME_BYTES
                and 0 <= age <= VOLUME_SNAPSHOT_TTL_SECONDS
            ):
                raise ValueError("Event volume snapshot is absent, stale, or belongs elsewhere")
            declared_free = (
                EVENT_VOLUME_BYTES
                - int(snapshot["used_bytes"])
                - OUTPUT_LEDGER_RESERVE_BYTES
                - INFLIGHT_RESERVE_BYTES
            )
            usable_disk = min(disk, declared_free)
            overcommit = int(Path("/proc/sys/vm/overcommit_memory").read_text())
            strict_commit = (
                int(mem["CommitLimit"] - mem["Committed_AS"]) if overcommit == 2 else None
            )
            allowed = bool(
                available >= 2 * GIB
                and (strict_commit is None or strict_commit >= 2 * GIB)
                and usable_disk - checkpoint_reservation >= 10_000_000_000
            )
            return allowed, {
                "schema": "ejepa_event_resources_v1",
                "host_available_bytes": available,
                "host_memavailable_raw_bytes": int(mem["MemAvailable"]),
                "windows_commit_headroom_bytes": None,
                "linux_cgroup_headroom_bytes": int(headroom),
                "cgroups": reports,
                "linux_strict_commit_headroom_bytes": strict_commit,
                "linux_overcommit_mode": overcommit,
                "filesystem_free_bytes": disk,
                "declared_event_volume_bytes": EVENT_VOLUME_BYTES,
                "volume_used_bytes": int(snapshot["used_bytes"]),
                "volume_snapshot_age_seconds": age,
                "volume_snapshot_ttl_seconds": VOLUME_SNAPSHOT_TTL_SECONDS,
                "output_ledger_reserve_bytes": OUTPUT_LEDGER_RESERVE_BYTES,
                "extra_inflight_reserve_bytes": INFLIGHT_RESERVE_BYTES,
                "disk_free_bytes": usable_disk,
                "disk_free_after_reservation_bytes": usable_disk - checkpoint_reservation,
                "physical_reserve_bytes": 2 * GIB,
                "completed_updates": completed_updates,
                "pod_memory_budget_bytes": self.budget_bytes,
                "memory_scope": "entire cgroup; charged cache included",
                "recursive_volume_scan_per_update": False,
            }
        except (OSError, ValueError, KeyError, RuntimeError, IndexError) as error:
            return False, {
                "schema": "ejepa_event_resources_v1",
                "error": f"{type(error).__name__}: {error}",
                "host_available_bytes": 0,
                "windows_commit_headroom_bytes": None,
            }
