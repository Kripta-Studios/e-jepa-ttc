"""Enrich only automatic pre-head EVENT resource pauses for safe supervision."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def install() -> None:
    from operational.rgb_port import accounting

    original: Callable[[Path, dict[str, Any]], None] = accounting.atomic_write_json
    if getattr(original, "_event_queue_pause_v1", False):
        return

    def write(path: Path, value: dict[str, Any]) -> None:
        payload = dict(value)
        if (
            path.name.startswith("EVENT_DOWNSTREAM_PAUSE_R")
            and payload.get("schema") == "ejepa_event_downstream_pause_v1"
            and payload.get("status") == "PAUSED_RECOVERABLE"
            and payload.get("reason") == "resource reserve unavailable"
        ):
            snapshot_path = path.parent / "RESOURCE_SNAPSHOT.json"
            if snapshot_path.is_file():
                resources = json.loads(snapshot_path.read_text(encoding="utf-8"))
            else:
                cgroup = Path("/sys/fs/cgroup/memory")
                current = int((cgroup / "memory.usage_in_bytes").read_text())
                limit = int((cgroup / "memory.limit_in_bytes").read_text())
                resources = {
                    "schema": "ejepa_event_queue_pause_resource_observation_v1",
                    "observed_timestamp": time.time(),
                    "cgroup_path": str(cgroup),
                    "current": current,
                    "effective_limit": limit,
                    "headroom": limit - current,
                    "source": "captured_at_automatic_pause_write",
                }
            stat = Path("/proc/self/stat").read_text(encoding="utf-8")
            tail = stat[stat.rindex(")") + 2 :].split()
            payload.update(
                manual=False,
                pause_classification="AUTOMATIC_RESOURCE_PAUSE",
                owner_pid=os.getpid(),
                owner_start_ticks=int(tail[19]),
                resources=resources,
            )
        original(path, payload)

    write._event_queue_pause_v1 = True  # type: ignore[attr-defined]
    accounting.atomic_write_json = write
