"""Selective retry after a zero-update deterministic fusion failure."""

from __future__ import annotations

import copy
from typing import Any


def retry_state(
    state: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Retain completed work, attempt counts and unrelated failures."""
    if any(item["status"] == "RUNNING" for item in state["tasks"].values()):
        raise ValueError("Recovery requires no running task")
    roots = {"F_TRUE", "F_ZERO"}
    for name in roots:
        item = state["tasks"][name]
        if item["status"] != "FAILED" or item.get("completed_updates") != 0:
            raise ValueError("Expected zero-update fusion failures")
    affected = set(roots)
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
        if name not in roots and record["status"] != "SKIPPED_DEPENDENCY":
            continue
        record["status"] = "WAITING"
        for key in ("reason", "retry_after_unix", "child"):
            record.pop(key, None)
        changed.append(name)
    result["campaign_status"] = "INCOMPLETE"
    return result, changed
