"""Explicit Linux cgroup v1/v2 resource policy for the versioned cloud overlay."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

GIB = 1024**3
DEFAULT_BUDGET_BYTES = 46_000_000_000
WORKSPACE_QUOTA_BYTES = 50_000_000_000


def storage_available(run_root: Path) -> tuple[int, dict[str, Any]]:
    """Bound shared-filesystem df by the user's confirmed 50 GB volume allocation."""
    free = shutil.disk_usage(run_root).free
    report: dict[str, Any] = {"filesystem_reported_free_bytes": free}
    if run_root.resolve().is_relative_to(Path("/workspace")):
        measured = subprocess.run(
            ["du", "-sb", "/workspace"], check=True, capture_output=True, text=True, timeout=60
        )
        used = int(measured.stdout.split()[0])
        if used < 0:
            raise ValueError("Invalid volume usage measurement")
        # Logical bytes are conservative for compressed files and include all
        # files under this user's mounted volume; symlinks are not followed.
        free = min(free, max(0, WORKSPACE_QUOTA_BYTES - used))
        report.update(
            user_confirmed_quota_bytes=WORKSPACE_QUOTA_BYTES,
            workspace_logical_bytes=used,
            quota_headroom_bytes=max(0, WORKSPACE_QUOTA_BYTES - used),
            capacity_source="user 50 GB allocation; never shared filesystem df alone",
        )
    return free, report


def memory_snapshot(
    root: Path, membership: str, budget_bytes: int
) -> tuple[int, list[dict[str, Any]]]:
    """Read actual charged memory and every accessible limiting ancestor."""
    if budget_bytes <= 2 * GIB:
        raise ValueError("Memory budget must exceed the 2 GiB reserve")
    entries = [line.split(":", 2) for line in membership.splitlines()]
    if any(len(entry) != 3 for entry in entries):
        raise ValueError("Malformed cgroup membership")
    v1 = [entry[2] for entry in entries if "memory" in entry[1].split(",")]
    v2 = [entry[2] for entry in entries if entry[:2] == ["0", ""]]
    if len(v1) > 1 or len(v2) > 1 or not (v1 or v2):
        raise ValueError("Missing or ambiguous cgroup memory controller")
    version, relative = (1, v1[0]) if v1 else (2, v2[0])
    parts = PurePosixPath(relative)
    if not parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Unsafe cgroup membership path")
    base = root / "memory" if version == 1 else root
    leaf = base / relative.lstrip("/")
    namespace_root_fallback = not leaf.is_dir()
    if namespace_root_fallback:
        # A container can expose its controller at the mount root while reporting
        # the host-relative membership. The budget still bounds this entire scope.
        leaf = base
    if not leaf.is_dir() or not leaf.resolve().is_relative_to(base.resolve()):
        raise ValueError("Missing or escaped cgroup directory")
    current_name = "memory.usage_in_bytes" if version == 1 else "memory.current"
    limit_name = "memory.limit_in_bytes" if version == 1 else "memory.max"
    reports = []
    path = leaf
    while True:
        if path == base and path != leaf and not (path / current_name).exists():
            break  # A true cgroup-v2 hierarchy root has no memory counters.
        current = int((path / current_name).read_text().strip())
        raw_limit = (path / limit_name).read_text().strip()
        limit = 2**63 - 1 if raw_limit == "max" and version == 2 else int(raw_limit)
        if current < 0 or limit <= 0:
            raise ValueError("Invalid cgroup memory counter")
        stats = {}
        for line in (path / "memory.stat").read_text().splitlines():
            key, value = line.split()
            stats[key] = int(value)
        if any(value < 0 for value in stats.values()):
            raise ValueError("Invalid memory.stat counter")
        if version == 1 and "hierarchical_memory_limit" in stats:
            if stats["hierarchical_memory_limit"] <= 0:
                raise ValueError("Invalid hierarchical memory limit")
            limit = min(limit, stats["hierarchical_memory_limit"])
        effective = min(limit, budget_bytes) if path == leaf else limit
        reports.append(
            {
                "path": str(path),
                "version": version,
                "current": current,
                "kernel_limit": limit,
                "effective_limit": effective,
                "headroom": max(0, effective - current),
                "pod_scope": path == leaf,
                "namespace_root_fallback": namespace_root_fallback,
                "memory_stat": stats,
                "cache_subtracted_from_usage": False,
            }
        )
        if path == base:
            break
        path = path.parent
    return min(report["headroom"] for report in reports), reports


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse /proc/meminfo with explicit units."""
    result = {}
    for line in text.splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        if len(fields) > 2 or (len(fields) == 2 and fields[1] != "kB"):
            raise ValueError("Unexpected meminfo unit")
        result[key] = int(fields[0]) * (1024 if len(fields) == 2 else 1)
    return result


class LinuxResourceMonitor:
    """Keep deadline, real memory reserve and checkpoint disk reserve fail-closed."""

    deadline_timestamp: float | None = None

    def __init__(
        self, *args: object, budget_bytes: int = DEFAULT_BUDGET_BYTES, **kwargs: object
    ) -> None:
        del args, kwargs
        if os.name != "posix":
            raise RuntimeError("Linux resource policy cannot monitor Windows training")
        if budget_bytes <= 2 * GIB:
            raise ValueError("Memory budget must exceed the 2 GiB reserve")
        self.budget_bytes = budget_bytes

    def guard(
        self, run_root: Path, *, checkpoint_reservation: int, completed_updates: int
    ) -> tuple[bool, dict[str, Any]]:
        """Count charged cache conservatively; never infer Pod RAM from host totals."""
        try:
            if self.deadline_timestamp is not None and time.time() >= self.deadline_timestamp:
                return False, {
                    "host_available_bytes": 0,
                    "windows_commit_headroom_bytes": None,
                    "reason": "CLOUD_EXECUTION_DEADLINE",
                    "pod_billing_stopped": False,
                }
            if checkpoint_reservation < 0 or completed_updates < 0:
                raise ValueError("Negative resource reservation or update count")
            mem = parse_meminfo(Path("/proc/meminfo").read_text())
            headroom, reports = memory_snapshot(
                Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), self.budget_bytes
            )
            available = min(mem["MemAvailable"], headroom)
            overcommit = int(Path("/proc/sys/vm/overcommit_memory").read_text())
            if overcommit not in {0, 1, 2}:
                raise ValueError("Unknown Linux overcommit mode")
            strict_commit = mem["CommitLimit"] - mem["Committed_AS"] if overcommit == 2 else None
            disk, storage = storage_available(run_root)
            allowed = (
                available >= 2 * GIB
                and (strict_commit is None or strict_commit >= 2 * GIB)
                and disk - checkpoint_reservation >= 10_000_000_000
            )
            return allowed, {
                "schema": "ejepa_linux_resources_v3",
                "host_available_bytes": available,
                "host_memavailable_raw_bytes": mem["MemAvailable"],
                "windows_commit_headroom_bytes": None,
                "linux_cgroup_headroom_bytes": headroom,
                "cgroups": reports,
                "linux_strict_commit_headroom_bytes": strict_commit,
                "linux_overcommit_mode": overcommit,
                "disk_free_bytes": disk,
                "disk_free_after_reservation_bytes": disk - checkpoint_reservation,
                "physical_reserve_bytes": 2 * GIB,
                "completed_updates": completed_updates,
                "pod_memory_budget_bytes": self.budget_bytes,
                "memory_scope": "entire cgroup; charged cache included without deduction",
                "fixed_23gb_usage_cap": False,
                "storage_allocation": storage,
            }
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            IndexError,
            subprocess.SubprocessError,
        ) as error:
            return False, {
                "schema": "ejepa_linux_resources_v3",
                "error": str(error),
                "host_available_bytes": 0,
                "windows_commit_headroom_bytes": None,
            }
