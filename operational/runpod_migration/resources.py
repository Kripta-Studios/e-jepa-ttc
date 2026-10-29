"""Fail-closed Linux cgroup-v2 guard for one isolated GPU worker."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

GIB = 1024**3


def parse_meminfo(text: str) -> dict[str, int]:
    """Read Linux /proc/meminfo byte counters."""
    result = {}
    for line in text.splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        result[key] = int(fields[0]) * (1024 if fields[1:] == ["kB"] else 1)
    return result


def cgroup_headroom(root: Path, current_path: str, budget_bytes: int) -> tuple[int, list[dict]]:
    """Use every readable ancestor limit plus the explicitly reserved Pod budget."""
    relative = next(
        (line.split(":", 2)[2] for line in current_path.splitlines() if line.startswith("0::")),
        None,
    )
    if relative is None:
        raise RuntimeError("cgroup v2 is required; v1 has not been validated")
    if ".." in Path(relative).parts:
        raise ValueError("Unsafe cgroup path")
    candidates = [root]
    nested = root / relative.lstrip("/")
    if nested != root and nested.is_dir():
        while nested != root:
            candidates.append(nested)
            nested = nested.parent
    leaf = max(candidates, key=lambda path: len(path.parts))
    reports = []
    for path in candidates:
        if path == root and leaf != root and not (path / "memory.current").exists():
            # The actual hierarchy root has no memory controller counter/limit.
            # Its children still must be readable; host MemAvailable is checked separately.
            continue
        current = int((path / "memory.current").read_text().strip())
        maximum = (path / "memory.max").read_text().strip()
        # Ancestor cgroups can include other Pods: apply our allocation only to
        # the leaf, while retaining any real finite ancestor limits.
        limit = (2**63 - 1) if maximum == "max" else int(maximum)
        if path == leaf:
            limit = min(limit, budget_bytes)
        if current < 0 or limit <= 0:
            raise ValueError("Invalid cgroup memory counters")
        reports.append(
            {
                "path": str(path),
                "current": current,
                "effective_limit": limit,
                "headroom": max(0, limit - current),
                "pod_scope": path == leaf,
            }
        )
    return min(x["headroom"] for x in reports), reports


class LinuxResourceMonitor:
    """Check actual container RAM and output filesystem on every optimizer boundary."""

    budget_bytes: int
    deadline_timestamp: float | None = None

    def __init__(self, *args: object, budget_bytes: int = 31_000_000_000, **kwargs: object) -> None:
        del args, kwargs
        if os.name != "posix":
            raise RuntimeError("LinuxResourceMonitor must run on Linux")
        self.budget_bytes = budget_bytes

    def guard(
        self, run_root: Path, *, checkpoint_reservation: int, completed_updates: int
    ) -> tuple[bool, dict[str, Any]]:
        """Unknown counters deny execution; no Windows commit counter is fabricated."""
        try:
            if self.deadline_timestamp is not None and time.time() >= self.deadline_timestamp:
                return False, {
                    "host_available_bytes": 0,
                    "windows_commit_headroom_bytes": None,
                    "reason": "CLOUD_EXECUTION_DEADLINE",
                    "pod_billing_stopped": False,
                }
            mem = parse_meminfo(Path("/proc/meminfo").read_text())
            headroom, reports = cgroup_headroom(
                Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), self.budget_bytes
            )
            available = min(mem["MemAvailable"], headroom)
            disk = shutil.disk_usage(run_root).free
            overcommit = int(Path("/proc/sys/vm/overcommit_memory").read_text())
            strict_commit = mem["CommitLimit"] - mem["Committed_AS"] if overcommit == 2 else None
            allowed = (
                available >= 2 * GIB
                and (strict_commit is None or strict_commit >= 2 * GIB)
                and disk - checkpoint_reservation >= 10_000_000_000
                and next(x["current"] for x in reports if x["pod_scope"]) <= 23_000_000_000
            )
            return allowed, {
                "schema": "ejepa_linux_resources_v1",
                "host_available_bytes": available,
                "windows_commit_headroom_bytes": None,
                "linux_cgroup_headroom_bytes": headroom,
                "linux_strict_commit_headroom_bytes": strict_commit,
                "linux_overcommit_mode": overcommit,
                "cgroups": reports,
                "disk_free_bytes": disk,
                "disk_free_after_reservation_bytes": disk - checkpoint_reservation,
                "completed_updates": completed_updates,
                "physical_reserve_bytes": 2 * GIB,
                "commit_policy": "cgroup plus host strict commit only when overcommit=2",
                "memory_scope": "all processes and charged cache in visible cgroup ancestors",
            }
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            return False, {
                "schema": "ejepa_linux_resources_v1",
                "error": str(error),
                "host_available_bytes": 0,
                "windows_commit_headroom_bytes": None,
            }
