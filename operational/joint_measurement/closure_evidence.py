"""Verify the bounded negative V18 programme before packaging; delivery remains separate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from operational.range_transfer.common import digest, read, sha, utc, write

from .bundle import require_quiescent
from .common import ROOT, RUN, accounting

STAGES = {
    "primary_evidence/COMPLETE.json": "PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING",
    "rgb/evidence/COMPLETE.json": "RGB_READY_FOR_INTERPRETATION",
    "fusion/evaluation/RESULT.json": "FUSION_METRICS_READY_FOR_INTERPRETATION",
    "replication/fusion_C1/evaluation/RESULT.json": "REPLICA_METRICS_READY_FOR_INTERPRETATION",
    **{
        f"adaptive/{folder}/evidence/COMPLETE.json": f"{family}_READY_FOR_INTERPRETATION"
        for family, folder in (
            ("C1", "C1_geometry"),
            ("D1", "D1_predictor_width"),
            ("B1", "B1_correspondence"),
            ("A1", "A1_adaptation_depth"),
        )
    },
    "cost/primary_raw_e2e/COMPLETE.json": "COMPLETE",
    "cost/garl_routes/RESULT.json": "COMPLETE",
}
ROUTES = {"FUSION-TRUE-S7", "FUSION-ZERO-S7", "R_CTX", "F_TRUE"}


def validate_cost_rows(registration: dict, rows: list[dict]) -> dict:
    """Recompute cost parity and population checks, without trusting saved booleans."""
    indices = registration["indices"]
    if len(indices) != 64 or len(set(indices)) != 64 or len(rows) != 64:
        raise ValueError("All 64 original paired cost queries are required")
    if registration["parity_rtol"] != 0 or registration["parity_atol"] != 0.001:
        raise ValueError("The disclosed absolute millisecond acceptance changed")
    strict_failures = 0
    maximum_difference = 0.0
    for index, row in zip(indices, rows, strict=True):
        if row["binding"] != digest(registration) or row["index"] != index:
            raise ValueError("Cost query binding/order changed")
        if set(row["timings"]) != ROUTES:
            raise ValueError("Every paired cost route is required")
        for timing in row["timings"].values():
            prediction, reference = timing["prediction"], timing["reference"]
            valid = all(math.isfinite(v) and (v < 0 or v > 0.1) for v in (prediction, reference))
            difference = abs(prediction - reference)
            strict = valid and difference <= 1e-4 + 1e-5 * abs(reference)
            if not (
                valid
                and (prediction < 0) == (reference < 0)
                and difference <= 0.001
                and timing["parity"]
                and timing["same_sign"]
                and timing["valid_native_outputs"]
                and timing["strict_parity"] == strict
                and math.isfinite(timing["seconds"])
                and timing["seconds"] > 0
                and timing["absolute_difference"] == difference
            ):
                raise ValueError("Invalid cost output, time or amended parity claim")
            strict_failures += not strict
            maximum_difference = max(maximum_difference, difference)
    return {
        "queries": 64,
        "paired_outputs": 256,
        "strict_failures": strict_failures,
        "maximum_absolute_difference_seconds": maximum_difference,
        "amended_acceptance_pass": True,
    }


def verify_cost(root: Path) -> dict:
    """Bind complete timings to the disclosed amendment and preserved original evidence."""
    directory = root / "fusion/cost/raw_e2e_millisecond_v1"
    registration = read(directory / "REGISTRATION.json")
    acceptance = read(directory / "ACCEPTANCE_REGISTRATION.json")
    complete = read(directory / "COMPLETE.json")
    if (
        complete["status"] != "COMPLETE"
        or not complete["E2E"]
        or complete["system_promotion"]
        or complete["binding"] != digest(registration)
        or registration["acceptance_registration_sha256"]
        != sha(directory / "ACCEPTANCE_REGISTRATION.json")
        or any(registration[k] != acceptance[k] for k in ("indices", "tokens", "orders"))
        or set(complete["results"]) != ROUTES
        or not complete["tolerance_amended_after_diagnostic"]
    ):
        raise ValueError("Incomplete or changed raw fusion cost admission")
    rows = [read(directory / f"row_{i:03d}.json") for i in range(64)]
    check = validate_cost_rows(registration, rows)
    historical = root / "fusion/cost/raw_e2e_historical_shape_v1"
    for field, filename in (
        ("historical_runtime_registration_sha256", "RUNTIME_REGISTRATION.json"),
        ("historical_cost_registration_sha256", "REGISTRATION.json"),
        ("historical_failure_sha256", "PARITY_FAILURE.json"),
    ):
        if sha(historical / filename) != acceptance[field]:
            raise ValueError("Original strict cost evidence changed")
    for name in ROUTES:
        values = np.array([row["timings"][name]["seconds"] for row in rows], dtype=np.float64)
        calculated = {
            "n": len(values),
            "mean_seconds": float(values.mean()),
            "p50_seconds": float(np.quantile(values, 0.5)),
            "p95_seconds": float(np.quantile(values, 0.95)),
            "p99_seconds": float(np.quantile(values, 0.99)),
            "queries_per_second": float(1 / values.mean()),
        }
        if complete["results"][name] != calculated:
            raise ValueError("Published cost summaries do not match all paired rows")
    if complete["strict_failure_count"] != check["strict_failures"]:
        raise ValueError("Strict failures were hidden or recounted incorrectly")
    if complete["strict_parity_all"] != (check["strict_failures"] == 0):
        raise ValueError("Strict cost result changed")
    reused = 0
    for row in rows:
        if "reused_from" in row:
            source = (root / row["reused_from"]).resolve()
            if not source.is_relative_to(root.resolve()) or sha(source) != row["reused_sha256"]:
                raise ValueError("Reused historical cost fragment changed")
            reused += 1
    if reused != complete["reused_rows"]:
        raise ValueError("Reused measurements must remain distinguished from new observations")
    return {**check, "reused_queries": reused, "complete_sha256": sha(directory / "COMPLETE.json")}


def build() -> dict:
    """Require closed writers, complete scientific evidence and an explicit conditional decision."""
    require_quiescent(ROOT, RUN)
    evidence = {}
    for name, expected in STAGES.items():
        if read(RUN / name)["status"] != expected:
            raise ValueError(f"Required V18 stage incomplete: {name}")
        evidence[name] = sha(RUN / name)
    decision_path = RUN / "DECISION_AFTER_A1.json"
    decision = read(decision_path)
    if (
        decision["status"] != "NO_ADMITTED_RECIPE_AFTER_FOUR_ADAPTIVE_COMPARISONS"
        or decision["adaptive_families_completed"] != ["C1", "D1", "B1", "A1"]
        or decision["A1_selection"]["ranked"]
        or decision["final_refit"]["event_enabled"]
        or decision["final_refit"]["multimodal_enabled"]
        or decision["new_test12_export"]["enabled"]
        or decision["V_after_A1"]["enabled"]
        or decision["SOTA_confirmation"]
    ):
        raise ValueError("This closure only covers the explicitly rejected final recipes")
    for name, expected in decision["evidence"].items():
        if sha(RUN / name) != expected:
            raise ValueError(f"Conditional decision evidence changed: {name}")
    a1 = read(RUN / "adaptive/A1_adaptation_depth/evidence/COMPLETE.json")
    if decision["A1_selection"] != a1["selection"]:
        raise ValueError("Final A1 decision differs from evaluated endpoints")
    for relative in ("system_compare/RESULT.json", "rgb/system_compare/RESULT.json"):
        if read(RUN / relative)["system_promotion"]:
            raise ValueError("A promoted recipe requires a different conditional closure")
    evidence["DECISION_AFTER_A1.json"] = sha(decision_path)
    cost = verify_cost(RUN)
    evidence["fusion/cost/raw_e2e_millisecond_v1/COMPLETE.json"] = cost["complete_sha256"]
    counted = accounting()
    budget = read(RUN / "GPU_BUDGET.json")
    total = sum(row["charged_s"] for row in budget["entries"])
    if counted["pending_upper"] or total > budget["cap_s"]:
        raise ValueError("Pending optimization work or GPU budget exceeded")
    journals = {
        p.relative_to(RUN).as_posix(): {"sha256": sha(p), **read(p)}
        for p in sorted(RUN.glob("**/UPDATE_JOURNAL.json"))
    }
    result = {
        "schema": "v18_bounded_program_evidence_v1",
        "status": "SCIENTIFIC_PROGRAM_VERIFIED_DELIVERY_PENDING",
        "utc": utc(),
        "full_objective_complete": False,
        "system_promotion": False,
        "SOTA_confirmation": False,
        "evidence": evidence,
        "raw_fusion_cost": cost,
        "accounting": counted,
        "counted_journals": journals,
        "gpu_scope_charged_upper_s": total,
        "gpu_scope_cap_s": budget["cap_s"],
        "retained_crash_reservations": [
            row for row in budget["entries"] if row["status"] == "RESERVED"
        ],
        "cost_scope": (
            "Own scopes including loading/saving and retained crash upper bounds; not kernels"
        ),
        "excluded_ci_activity": (
            "Two unrelated repository CI CUDA optimizer updates occurred during the initial "
            "local CI test run; not V18 fits. They are disclosed separately, "
            "not hidden as CPU tests."
        ),
        "remaining_work": ["Verify the <=100000000-byte final archive and independent replay"],
        "delivery_boundary": (
            "This receipt precedes packaging. The external independent-replay receipt must bind "
            "the final ZIP; this file cannot certify an archive that does not yet exist."
        ),
    }
    write(RUN / "PROGRAM_EVIDENCE.json", result)
    return {k: v for k, v in result.items() if k not in ("counted_journals", "evidence")}


if __name__ == "__main__":
    print(build())
