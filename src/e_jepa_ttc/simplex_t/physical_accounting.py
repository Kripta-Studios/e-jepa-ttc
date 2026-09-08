"""Read-only reconstruction of optimizer-work journal accounting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def audit_physical_work(
    state: dict,
    *,
    expected_graph: dict[str, int],
    technical_reserved: int,
    resource_ok: Callable[[], bool],
) -> dict:
    """Reconstruct transitions rather than trusting a journal's summary counters.

    Caller must pin journal bytes and verify graph authority and checkpoint
    training states separately. Pending reservations and possible crash work
    are upper bounds, not observed optimizer updates or scientific completion.
    No journal transitions, file writes or checkpoint loads are performed here.
    """
    if (
        not expected_graph
        or len(expected_graph) > 84
        or any(type(n) is not int or n != 2500 for n in expected_graph.values())
        or type(technical_reserved) is not int
        or not 0 <= technical_reserved <= 1045
    ):
        raise ValueError("bounded registered graph and technical reservation required")
    if (
        state.get("schema") != "simplex_t_physical_work_v1"
        or state["graph"] != expected_graph
        or state["technical_reserved"] != technical_reserved
    ):
        raise ValueError("physical journal contract differs")
    reconstructed: dict[str, dict[str, Any]] = {}
    for event in state["events"]:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: physical work accounting")
        key, completed, operation = event["key"], event["completed"], event["operation"]
        if key not in expected_graph or type(completed) is not int or not 0 <= completed <= 2500:
            raise ValueError("invalid journal event identity or progress")
        fit = reconstructed.setdefault(key, dict(completed=0, uncertain_lost_upper=0, pending=None))
        if operation == "begin":
            if (
                fit["pending"] is not None
                or completed != fit["completed"]
                or completed == 2500
                or event["checkpoint_sha256"] is not None
            ):
                raise ValueError("invalid or overlapping physical work reservation")
            fit["pending"] = [completed, min((completed // 100 + 1) * 100, 2500)]
        elif operation in {"checkpoint", "recover"}:
            pending, digest = fit["pending"], event["checkpoint_sha256"]
            if (
                pending is None
                or not pending[0] <= completed <= pending[1]
                or not isinstance(digest, str)
                or len(digest) != 64
                or set(digest) - set("0123456789abcdef")
            ):
                raise ValueError("invalid physical checkpoint settlement")
            if operation == "recover":
                fit["uncertain_lost_upper"] += pending[1] - completed
            fit.update(completed=completed, pending=None, checkpoint_sha256=digest)
        else:
            raise ValueError("unknown physical journal operation")
    if set(state["fits"]) != set(reconstructed):
        raise ValueError("physical fit set differs from events")
    for key, fit in reconstructed.items():
        actual = dict(state["fits"][key])
        # This digest is not included in journal events. Never claim that replay
        # of events verifies it; the endpoint validator must check actual states.
        state_digest = actual.pop("checkpoint_state_sha256", None)
        if state_digest is not None and (
            not isinstance(state_digest, str)
            or len(state_digest) != 64
            or set(state_digest) - set("0123456789abcdef")
        ):
            raise ValueError("invalid checkpoint state digest")
        if actual != fit:
            raise ValueError("physical fit counters differ from reconstructed events")
    saved = sum(fit["completed"] for fit in reconstructed.values())
    lost = sum(fit["uncertain_lost_upper"] for fit in reconstructed.values())
    pending = sum(
        fit["pending"][1] - fit["pending"][0]
        for fit in reconstructed.values()
        if fit["pending"] is not None
    )
    planned_upper = sum(expected_graph.values()) + technical_reserved + lost
    expected_accounting = dict(
        scientific_saved_updates=saved,
        scientific_uncertain_lost_lower=0,
        scientific_uncertain_lost_upper=lost,
        technical_reserved_not_execution_claim=technical_reserved,
        full_graph_physical_work_upper=planned_upper,
    )
    if state["accounting"] != expected_accounting or planned_upper > 250000:
        raise ValueError("physical work summary differs or exceeds hard cap")
    return dict(
        status="JOURNAL_EVENTS_RECONCILED_NOT_CHECKPOINT_OR_SCIENTIFIC_COMPLETION",
        saved_updates=saved,
        possible_lost_updates_upper=lost,
        pending_updates_upper=pending,
        recorded_work_lower=saved,
        recorded_work_upper=saved + lost + pending,
        full_registered_graph_and_technical_upper=planned_upper,
        technical_reserved_not_execution_claim=technical_reserved,
        journal_endpoint_fits=sum(fit["completed"] == 2500 for fit in reconstructed.values()),
        checkpoint_training_states_verified=False,
        optimizer_updates_executed=0,
    )
