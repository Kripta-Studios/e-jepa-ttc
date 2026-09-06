"""Conservative accounting for optimizer work lost between durable checkpoints.

Unlike endpoint progress, physical work can include replayed updates. This
journal reserves at most100 updates before a chunk and retains uncertainty after
a crash. It never reports reserved or possibly lost work as observed execution.
The caller must hold the fit writer lease and validate checkpoint identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .lifecycle import ExclusiveLease


class WorkBudget:
    """Reserve the entire registered graph plus technical work and possible replay."""

    def __init__(self, path: Path, graph: dict[str, int], technical_reserved: int) -> None:
        if (
            not graph
            or len(graph) > 84
            or any(type(n) is not int or n != 2500 for n in graph.values())
        ):
            raise ValueError("invalid scientific graph")
        if type(technical_reserved) is not int or not 0 <= technical_reserved <= 1000:
            raise ValueError("invalid technical reservation")
        self.path, self.graph, self.technical_reserved = path, graph, technical_reserved

    def transition(
        self, operation: str, key: str, completed: int, *, checkpoint_sha256: str | None = None
    ) -> dict[str, Any]:
        """Begin a chunk, settle a safe checkpoint, or recover uncertain crashed work.

        `checkpoint` is only for synchronous publication immediately after the
        engine's save; `recover` is for a restarted process using its validated
        checkpoint. A recovery charges the unsaved suffix conservatively, even
        if the process might have died before executing it. Never auto-clear a
        surviving lease to reach this operation.
        """
        if key not in self.graph or type(completed) is not int or not 0 <= completed <= 2500:
            raise ValueError("invalid fit/progress")
        if operation not in {"begin", "checkpoint", "recover"}:
            raise ValueError("unknown work-budget operation")
        if operation != "begin" and (
            checkpoint_sha256 is None
            or len(checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
        ):
            raise ValueError("validated checkpoint byte identity required")
        with ExclusiveLease(self.path.with_suffix(".lock")):
            state = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {
                    "schema": "simplex_t_physical_work_v1",
                    "graph": self.graph,
                    "technical_reserved": self.technical_reserved,
                    "fits": {},
                    "events": [],
                }
            )
            if (
                state.get("schema") != "simplex_t_physical_work_v1"
                or state["graph"] != self.graph
                or state["technical_reserved"] != self.technical_reserved
            ):
                raise ValueError("physical-work contract changed")
            fits = state["fits"]
            fit = fits.setdefault(key, {"completed": 0, "uncertain_lost_upper": 0, "pending": None})
            if operation == "begin":
                if fit["pending"] is not None:
                    raise ValueError("unsettled chunk requires explicit checkpoint recovery")
                if completed != fit["completed"] or completed == 2500:
                    raise ValueError("cannot rewind, skip or rerun endpoint")
                fit["pending"] = [completed, min((completed // 100 + 1) * 100, 2500)]
            else:
                pending = fit["pending"]
                if pending is None or not pending[0] <= completed <= pending[1]:
                    raise ValueError("checkpoint outside reserved chunk")
                if operation == "recover":
                    fit["uncertain_lost_upper"] += pending[1] - completed
                fit.update(completed=completed, pending=None, checkpoint_sha256=checkpoint_sha256)
            lost_upper = sum(f["uncertain_lost_upper"] for f in fits.values())
            required = sum(self.graph.values()) + self.technical_reserved + lost_upper
            if required > 250000:
                raise ValueError("physical optimizer work upper bound exceeds250000")
            state["events"].append(
                {
                    "operation": operation,
                    "key": key,
                    "completed": completed,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
            state["accounting"] = {
                "scientific_saved_updates": sum(f["completed"] for f in fits.values()),
                "scientific_uncertain_lost_lower": 0,
                "scientific_uncertain_lost_upper": lost_upper,
                "technical_reserved_not_execution_claim": self.technical_reserved,
                "full_graph_physical_work_upper": required,
            }
            atomic_json(self.path, state)
            return state


class EngineWorkJournal:
    """Bridge a fit's safe callbacks to its persistent physical-work journal.

    The engine validates checkpoint training identity before calling start.
    A fit writer lease must cover the entire engine invocation; transaction
    leases here protect the shared budget only, not concurrent model updates.
    """

    def __init__(self, budget: WorkBudget, key: str) -> None:
        if key not in budget.graph:
            raise ValueError("unknown fit")
        self.budget, self.key = budget, key
        self.active_end: int | None = None
        self.saved = 0
        self.started = False

    def start(self, completed: int, checkpoint: Path) -> None:
        """Reconcile a validated resume checkpoint without dropping uncertain work."""
        if self.started or not checkpoint.is_file():
            raise ValueError("journal requires one start and an existing checkpoint")
        fit = {"completed": 0, "pending": None}
        if self.budget.path.exists():
            state = json.loads(self.budget.path.read_text(encoding="utf-8"))
            if (
                state["graph"] != self.budget.graph
                or state["technical_reserved"] != self.budget.technical_reserved
            ):
                raise ValueError("physical-work contract changed")
            fit = state["fits"].get(self.key, fit)
        if fit["pending"] is not None:
            self.budget.transition(
                "recover", self.key, completed, checkpoint_sha256=sha256(checkpoint)
            )
        elif fit["completed"] != completed:
            raise ValueError("checkpoint and settled journal progress disagree")
        self.saved, self.started = completed, True

    def before_update(self, completed: int) -> None:
        """Persist a chunk reservation before its first optimizer update."""
        if not self.started:
            raise ValueError("journal not started")
        if self.active_end is None:
            if completed != self.saved:
                raise ValueError("unsaved progress without a reservation")
            state = self.budget.transition("begin", self.key, completed)
            end = state["fits"][self.key]["pending"][1]
            if type(end) is not int:
                raise ValueError("journal reservation endpoint must be an integer")
            self.active_end = end
        if self.active_end is None or not self.saved <= completed < self.active_end:
            raise ValueError("update outside journal reservation")

    def checkpoint_saved(self, completed: int, checkpoint: Path) -> None:
        """Settle only after atomic checkpoint publication has returned successfully."""
        if not self.started:
            raise ValueError("journal not started")
        if self.active_end is None:
            if completed != self.saved:
                raise ValueError("checkpoint progress without a reservation")
            return
        self.budget.transition(
            "checkpoint", self.key, completed, checkpoint_sha256=sha256(checkpoint)
        )
        self.saved, self.active_end = completed, None
