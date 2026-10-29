"""User-authorized one-millisecond cost equivalence; retain original strict results."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from unittest.mock import patch

from operational.range_transfer.common import digest, freeze, read, sha, write

from . import fusion_cost
from . import fusion_cost_historical_shape as historical
from .common import ROOT, RUN
from .compute_guard import guard_idle
from .fusion_cost_native import admit, native_context, qa
from .fusion_cost_shape_probe import ShapeRuntime
from .train import committed_sources

OUT = RUN / "fusion/cost/raw_e2e_millisecond_v1"
LIMIT_SECONDS = 0.001


def equivalence(value: float, reference: float) -> dict:
    """Accept finite, same-sign native outputs within an absolute millisecond."""
    valid = all(math.isfinite(x) and (x < 0 or x > 0.1) for x in (value, reference))
    same_sign = valid and (value < 0) == (reference < 0)
    difference = abs(value - reference)
    strict = valid and difference <= 1e-4 + 1e-5 * abs(reference)
    return {
        "strict_parity": strict,
        "valid_native_outputs": valid,
        "same_sign": same_sign,
        "parity": bool(same_sign and difference <= LIMIT_SECONDS),
        "absolute_difference": difference,
    }


def annotate(row: dict) -> dict:
    """Retain measured predictions and timings, adding both numerical decisions."""
    result = copy.deepcopy(row)
    if set(result["timings"]) != set(fusion_cost.NAMES):
        raise ValueError("All four paired routes are required")
    for timing in result["timings"].values():
        timing.update(equivalence(timing["prediction"], timing["reference"]))
    return result


def register() -> dict:
    """Record the post-diagnostic authorization before any additional cost measurement."""
    old = read(historical.OUT / "RUNTIME_REGISTRATION.json")
    if old["sources"] != historical.sources():
        raise ValueError("Historical cost sources changed")
    inventory = read(historical.INVENTORY / "RESULT.json")
    historical.shape_map(inventory, old["indices"])
    source = "operational/joint_measurement/fusion_cost_millisecond.py"
    sources = {**historical.sources(), source: sha(ROOT / source)}
    committed_sources(sources)
    record = {
        "schema": "v18_user_authorized_cost_equivalence_v1",
        "sources": sources,
        "authorization": (
            "User: esa tolerancia del 0,00214 % de la predicción me parece lo "
            "suficinetemente pequeña como para que lo ignoremos"
        ),
        "interpretation": (
            "Assistant operationalizes acceptance as <=0.001 seconds absolute for every "
            "route/query, finite native outputs and unchanged sign. User accepted the observed "
            "0.00079345703125-second discrepancy; this is a disclosed post-diagnostic change, "
            "not an originally preregistered tolerance or a relaxation of scientific gates."
        ),
        "historical_runtime_registration_sha256": sha(historical.OUT / "RUNTIME_REGISTRATION.json"),
        "historical_cost_registration_sha256": sha(historical.OUT / "REGISTRATION.json"),
        "historical_failure_sha256": sha(historical.OUT / "PARITY_FAILURE.json"),
        "inventory_sha256": sha(historical.INVENTORY / "RESULT.json"),
        "indices": old["indices"],
        "tokens": old["tokens"],
        "orders": old["orders"],
        "strict_tolerance": {"rtol": 1e-5, "atol": 1e-4},
        "accepted_tolerance": {"rtol": 0.0, "atol": LIMIT_SECONDS},
        "reuse": (
            "Keep original measured timings for available immutable historical fragments; "
            "bind each source hash, retain strict failures and all queries, measure missing rows."
        ),
        "optimizer_updates": 0,
        "scientific_metrics_changed": False,
        "system_promotion": False,
    }
    if len(set(record["indices"])) != len(record["indices"]) or len(set(record["tokens"])) != len(
        record["tokens"]
    ):
        raise ValueError("Duplicate native cost queries")
    freeze(OUT / "ACCEPTANCE_REGISTRATION.json", record)
    return record


def prior_rows(record: dict) -> list[tuple[Path, dict]]:
    """Check original fragment identity and complete paired coverage before reuse."""
    old_binding = digest(read(historical.OUT / "REGISTRATION.json"))
    rows = []
    for number, index in enumerate(record["indices"]):
        path = historical.OUT / f"row_{number:03d}.json"
        if not path.exists():
            continue
        row = read(path)
        if row["binding"] != old_binding or row["index"] != index:
            raise ValueError("Historical fragment identity/position changed")
        rows.append((path, annotate(row)))
    return rows


def audit() -> dict:
    """Evaluate saved outputs only: no GPU, GT, new predictions or timing measurements."""
    record = register()
    rows = prior_rows(record)
    result = {
        "registration_sha256": sha(OUT / "ACCEPTANCE_REGISTRATION.json"),
        "status": "PARTIAL_SAVED_OUTPUT_AUDIT",
        "rows_available": len(rows),
        "rows_required": len(record["indices"]),
        "accepted": all(t["parity"] for _, r in rows for t in r["timings"].values()),
        "strict_failures": [
            {"index": r["index"], "route": name, **t}
            for _, r in rows
            for name, t in r["timings"].items()
            if not t["strict_parity"]
        ],
        "source_fragments": {p.name: sha(p) for p, _ in rows},
        "cost_complete": False,
        "optimizer_updates": 0,
    }
    freeze(OUT / "SAVED_OUTPUT_AUDIT.json", result)
    return result


def run() -> dict:
    """Continue the full unchanged population once exclusive GPU ownership is available."""
    guard_idle()
    record = register()
    prior = prior_rows(record)
    if not all(t["parity"] for _, row in prior for t in row["timings"].values()):
        raise ValueError("Saved output exceeds the authorized millisecond acceptance")
    mapping = historical.shape_map(read(historical.INVENTORY / "RESULT.json"), record["indices"])
    native = admit()
    qa(native)

    class MillisecondRuntime(ShapeRuntime):
        convolution_tf32 = True
        shape_map = mapping

    def bound_freeze(path: Path, value: dict) -> None:
        if path == OUT / "REGISTRATION.json":
            if any(value[k] != record[k] for k in ("indices", "tokens", "orders")):
                raise ValueError("Original population or paired order changed")
            historical.bind_cost(
                value,
                record["historical_runtime_registration_sha256"],
                record["inventory_sha256"],
            )
            value.update(
                acceptance_registration_sha256=sha(OUT / "ACCEPTANCE_REGISTRATION.json"),
                parity_rtol=0.0,
                parity_atol=LIMIT_SECONDS,
                acceptance_scope="Post-diagnostic user-authorized absolute 1ms; same sign/domain",
            )
            freeze(path, value)
            for original, row in prior:
                row.update(
                    binding=digest(value),
                    reused_from=str(original.relative_to(RUN)),
                    reused_sha256=sha(original),
                )
                freeze(OUT / original.name, row)
            return
        if path.name.startswith("row_"):
            # measure() tests this same dict after freezing; reject sign/domain drift too.
            value.update(annotate(value))
        if path.name == "COMPLETE.json":
            rows = [read(OUT / f"row_{i:03d}.json") for i in range(len(record["indices"]))]
            value.update(
                parity="All selected outputs within user-authorized absolute 1ms; same sign/domain",
                strict_parity_all=all(
                    t["strict_parity"] for r in rows for t in r["timings"].values()
                ),
                strict_failure_count=sum(
                    not t["strict_parity"] for r in rows for t in r["timings"].values()
                ),
                reused_rows=len(prior),
                tolerance_amended_after_diagnostic=True,
            )
        freeze(path, value)

    with (
        native_context(native),
        patch.object(fusion_cost, "OUT", OUT),
        patch.object(fusion_cost, "RouteRuntime", MillisecondRuntime),
        patch.object(fusion_cost, "freeze", bound_freeze),
    ):
        result = fusion_cost.measure()
    write(OUT / "LATEST.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "audit", "run"))
    mode = parser.parse_args().mode
    result = {"register": register, "audit": audit, "run": run}[mode]()
    print({key: result.get(key) for key in ("status", "rows_available", "accepted", "next_row")})
