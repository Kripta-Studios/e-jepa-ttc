"""Keep the full resource policy while avoiding unrelated-process RSS queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil

from operational.rgb_port import train_producers as training


def resource_guard(run_root: Path, *, checkpoint_reservation: int) -> tuple[bool, dict[str, Any]]:
    """Apply the original policy; query memory only after the original filter.

    Every call obtains fresh process commands and membership. ``as_dict`` keeps
    the original psutil handling of inaccessible attributes. The caller applies
    the separately admitted commit-reserve overlay after this historical policy.
    The calling cadence, fast guard, and resource thresholds remain unchanged.
    """
    repository = Path(training.__file__).resolve().parents[2]
    project_markers = (str(repository).lower(), repository.name.lower(), "operational.rgb_port")
    aggregate = 0
    project_processes: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if any(marker in command for marker in project_markers) and "pytest" not in command:
                rss = int(process.as_dict(attrs=["memory_info"])["memory_info"].rss)
                aggregate += rss
                project_processes.append({"pid": process.pid, "rss": rss})
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    commit = training._windows_commit_headroom()
    disk_free = int(psutil.disk_usage(str(run_root.resolve().anchor)).free)
    values = {
        "host_available_bytes": int(psutil.virtual_memory().available),
        "project_tree_rss_bytes": aggregate,
        "project_processes": project_processes,
        "windows_commit_headroom_bytes": commit,
        "disk_free_bytes": disk_free,
        "checkpoint_reservation_bytes": checkpoint_reservation,
        "disk_free_after_reservation_bytes": disk_free - checkpoint_reservation,
    }
    allowed = (
        values["host_available_bytes"] >= 2 * 1024**3
        and values["project_tree_rss_bytes"] <= 23_000_000_000
        and (commit is None or commit >= 3 * 1024**3)
        and values["disk_free_after_reservation_bytes"] >= 10_000_000_000
    )
    return allowed, values
