"""Prepare a bounded retry of H_R_RAW and its skipped dependency descendants."""

from __future__ import annotations

import copy
from typing import Any


def retry_state(
    state: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Retain completed tasks and attempt accounting; never reset unrelated failures."""
    if any(item["status"] == "RUNNING" for item in state["tasks"].values()):
        raise ValueError("Recovery requires no running task")
    root = "H_R_RAW"
    if state["tasks"][root]["status"] != "FAILED":
        raise ValueError("Expected the recorded H_R_RAW failure")
    affected = {root}
    while True:
        expanded = affected | {
            task["id"] for task in tasks if affected.intersection(task.get("depends", []))
        }
        if expanded == affected:
            break
        affected = expanded
    result = copy.deepcopy(state)
    changed = []
    for name in sorted(affected):
        record = result["tasks"][name]
        if name != root and record["status"] != "SKIPPED_DEPENDENCY":
            continue
        record["status"] = "WAITING"
        record.pop("reason", None)
        record.pop("retry_after_unix", None)
        record.pop("child", None)
        changed.append(name)
    result["campaign_status"] = "INCOMPLETE"
    return result, changed
