"""Absolute resource admission and an exclusive, bounded optimizer-update ledger."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Literal

import psutil

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json
from e_jepa_ttc.artifacts.simplex_t_preflight import resource_headroom

TECHNICAL_UPDATE_CAP = 1125


class ExclusiveLease:
    """Never steal a stale-looking lease or remove another owner's file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> ExclusiveLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"owner": self.owner, "pid": os.getpid(), "host": socket.gethostname()}, stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True
        return self

    def __exit__(self, *exception: object) -> Literal[False]:
        if self.acquired:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state["owner"] != self.owner:
                raise RuntimeError("lease owner changed; refusing removal")
            self.path.unlink()
            self.acquired = False
        return False


def admitted(written_volumes: list[Path]) -> dict[str, Any]:
    """Read process-tree memory and exact free bytes; never ratio-based admission."""
    process = psutil.Process()
    try:
        rss = process.memory_info().rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
        available = psutil.virtual_memory().available
        free = [psutil.disk_usage(str(path)).free for path in written_volumes]
    except (psutil.Error, OSError) as error:
        return {"has_headroom": False, "reasons": [f"RESOURCE_COUNTER_UNAVAILABLE:{error}"]}
    return {
        **resource_headroom(
            available_ram=available, process_tree_rss=rss, written_volume_free=free
        ),
        "process_tree_rss_bytes": rss,
        "host_available_bytes": available,
        "written_volume_free_bytes": free,
    }


class TechnicalBudget:
    """Charge unique technical operations before execution, never refund crashes.

    Reservations are conservative upper bounds, not evidence of executed updates.
    The campaign must use one persistent path across output directories.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def reserve(self, key: str, updates: int) -> dict[str, Any]:
        """Refuse duplicates; the cap includes retained failures and consumer-binding QA."""
        if not key or type(updates) is not int or not 1 <= updates <= TECHNICAL_UPDATE_CAP:
            raise ValueError("invalid technical reservation")
        with ExclusiveLease(self.path.with_suffix(".lock")):
            state = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {"schema": "simplex_t_technical_budget_v1", "reservations": {}}
            )
            if state.get("schema") != "simplex_t_technical_budget_v1":
                raise ValueError("unrecognized technical budget schema")
            reservations = state["reservations"]
            if any(type(value) is not int or value < 1 for value in reservations.values()):
                raise ValueError("invalid saved technical reservation")
            if key in reservations:
                raise ValueError("technical operation already reserved; do not repeat")
            if sum(reservations.values()) + updates > TECHNICAL_UPDATE_CAP:
                raise ValueError("technical budget exceeded")
            reservations[key] = updates
            atomic_json(self.path, state)
            return state


class UpdateLedger:
    """Reserve only a frozen graph; persist counted progress under a writer lease."""

    def __init__(self, path: Path, graph: dict[str, int]) -> None:
        if any(updates != 2500 for updates in graph.values()) or len(graph) > 84:
            raise ValueError("unregistered scientific endpoint graph")
        self.path, self.graph = path, graph

    def transaction(self, operation: str, key: str, amount: int = 0) -> dict[str, Any]:
        """Technical increments and monotonic fit counters cannot create new arms."""
        if type(amount) is not int or amount < 0:
            raise ValueError("nonnegative integer updates required")
        with ExclusiveLease(self.path.with_suffix(".lock")):
            state = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {"graph": self.graph, "fits": {}, "technical_updates": 0}
            )
            if state["graph"] != self.graph:
                raise ValueError("cannot change frozen fit graph during resume")
            if operation == "technical":
                state["technical_updates"] += amount
            elif operation == "reserve":
                if key not in self.graph or key in state["fits"]:
                    raise ValueError("unknown or already-reserved fit")
                state["fits"][key] = {"updates": 0, "status": "RESERVED"}
            elif operation == "progress":
                if key not in state["fits"]:
                    raise ValueError("fit not reserved")
                fit = state["fits"][key]
                if amount < fit["updates"] or amount > 2500:
                    raise ValueError("nonmonotonic or excessive update endpoint")
                fit.update(updates=amount, status="COMPLETED" if amount == 2500 else "PARTIAL")
            else:
                raise ValueError("unknown ledger operation")
            reserved = len(state["fits"]) * 2500
            if state["technical_updates"] > TECHNICAL_UPDATE_CAP or reserved > 210000:
                raise ValueError("registered scientific/technical budget exceeded")
            if reserved + state["technical_updates"] > 250000:
                raise ValueError("absolute optimizer update cap exceeded")
            atomic_json(self.path, state)
            return state
