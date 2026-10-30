"""Bind existing PHASE/Garl outputs to query clocks before comparison."""

from __future__ import annotations

from typing import Any


def saved_predictions(
    ledger: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    """Retain missing outputs and reject silently relabeled row identities."""
    population = {t["query_id"]: t for t in targets}
    phase_keys = {"PHASE": "model_ttc", "PHASE_event": "event_ttc", "PHASE_rgb": "rgb_ttc"}
    if method not in {*phase_keys, "Garl_full", "Garl_event", "Garl_visual"}:
        raise ValueError("Unknown saved baseline")
    rows = []
    seen = set()
    for row in ledger:
        if method.startswith("Garl_") and row["variant"] != method.removeprefix("Garl_"):
            continue
        identity = row["identity"]
        qid = f"{identity['sequence_id']}:{identity['row_index']}"
        if qid in seen or qid not in population:
            raise ValueError("Duplicate or unknown baseline query")
        seen.add(qid)
        if identity["anchor_us"] != population[qid]["query_us"]:
            raise ValueError("Saved prediction clock differs from target query")
        value = (
            (row.get("prediction") or {}).get(phase_keys[method])
            if method in phase_keys
            else row["prediction_ttc_seconds"]
        )
        rows.append({"query_id": qid, "prediction": value})
    return rows
