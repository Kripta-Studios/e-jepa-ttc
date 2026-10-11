"""Selective retry of failed external routes and regeneration of their report."""

from __future__ import annotations

import copy
from typing import Any


def retry_state(
    state: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Retain trained endpoints and attempt counts; regenerate the derived report."""
    if any(item["status"] == "RUNNING" for item in state["tasks"].values()):
        raise ValueError("Recovery requires no running task")
    roots = {"DEV32_PREDICT", "PROFILE_ROUTES"}
    for name in roots:
        item = state["tasks"][name]
        if item["status"] != "FAILED" or item.get("completed_updates") != 0:
            raise ValueError("Expected zero-update external inference failures")
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
    report = result["tasks"].get("REPORT")
    if report is not None and report["status"] == "COMPLETE":
        report["status"] = "WAITING"
        report.pop("completion_sha256", None)
        changed.append("REPORT")
    result["campaign_status"] = "INCOMPLETE"
    return result, changed
