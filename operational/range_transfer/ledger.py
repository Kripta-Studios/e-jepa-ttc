"""Read and bound the durable update ledger without importing numerical libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read, write


def accounting(run: Path, *, prospective: int = 0, technical: bool = False) -> dict[str, Any]:
    """Count durable, executed, pending/uncertain and recovery across this campaign."""
    totals = {
        "scientific": 0,
        "technical": 0,
        "recovery_upper": 0,
        "pending_upper": 0,
        "durable": 0,
        "physical_upper": 0,
    }
    details = []
    for path in sorted(run.glob("fits/*/UPDATE_JOURNAL.json")) + sorted(
        run.glob("qa/*/UPDATE_JOURNAL.json")
    ):
        value = read(path)
        kind = "technical" if path.parent.parent.name == "qa" else "scientific"
        count = int(value["completed_updates"])
        recovery = int(value["recovery_upper"])
        pending = int(value["pending_update_upper"])
        totals[kind] += count
        totals["durable"] += int(value["durable_updates"])
        totals["recovery_upper"] += recovery
        totals["pending_upper"] += pending
        details.append({"fit": path.parent.name, "kind": kind, **value})
    totals["physical_upper"] = (
        totals["scientific"]
        + totals["technical"]
        + totals["recovery_upper"]
        + totals["pending_upper"]
    )
    caps = (
        (totals["scientific"] + (0 if technical else prospective), 25000),
        (totals["technical"] + (prospective if technical else 0), 500),
        (totals["recovery_upper"] + totals["pending_upper"], 4500),
        (totals["physical_upper"] + prospective, 30000),
    )
    if any(used > cap for used, cap in caps):
        raise RuntimeError(f"RANGE-TRANSFER budget exhausted: {totals}")
    result = {
        "schema": "range_transfer_accounting_v1",
        **totals,
        "fits": details,
        "caps": {"scientific": 25000, "technical": 500, "recovery": 4500, "physical": 30000},
        "savings_reassigned": False,
    }
    write(run / "ACCOUNTING.json", result)
    return result
