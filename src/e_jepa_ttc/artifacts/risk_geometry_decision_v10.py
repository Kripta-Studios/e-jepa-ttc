"""Fail-closed numerical branch decisions; production verifies file receipts first.

This module never repairs a historical decision, chooses the best control,
constructs statistics, or authorizes protected data. Pass comparisons with paired
candidate-minus-reference signs. Missing/non-finite statistics are blocks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class DecisionBlocked(ValueError):  # noqa: N818 -- frozen reference interface
    """Required current-campaign evidence is absent or invalid."""


def _number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionBlocked(f"missing/non-numeric statistic: {key}")
    value = float(value)
    if not math.isfinite(value):
        raise DecisionBlocked(f"non-finite statistic: {key}")
    return value


def _ready(evidence: Mapping[str, Any]) -> None:
    for key in (
        "execution_complete",
        "protocol_conformant",
        "diagnostics_complete",
        "coverage_unchanged",
    ):
        if evidence.get(key) is not True:
            raise DecisionBlocked(f"readiness not established: {key}")
    if _number(evidence, "finite_fraction") != 1.0 or _number(evidence, "failure_count") != 0:
        raise DecisionBlocked("non-finite predictions or nonzero failure count")


def _comparison(evidence: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    comparisons = evidence.get("comparisons")
    if not isinstance(comparisons, Mapping) or not isinstance(comparisons.get(reference), Mapping):
        raise DecisionBlocked(f"missing comparison to {reference}")
    return comparisons[reference]


def _utility(record: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    # Evaluate all fields before all(): no missing evidence hidden by short-circuit.
    checks = [
        _number(record, "point_delta") <= rule["point_delta_lte"],
        _number(record, "hierarchical_ci95_high") < rule["hierarchical_ci95_high_lt"],
        _number(record, "sequence_ci95_high") < rule["sequence_ci95_high_lt"],
        _number(record, "weighted_sign_error_delta") <= rule["weighted_sign_error_delta_lte"],
        _number(record, "crucial_bucket_mid_delta") <= rule["crucial_bucket_mid_delta_lte"],
    ]
    return all(checks)


def _mechanism(record: Mapping[str, Any], rule: Mapping[str, Any], high: float = 0.0) -> bool:
    delta = _number(record, "point_delta")
    point = (
        delta <= rule["point_delta_lte"]
        if "point_delta_lte" in rule
        else delta < rule["point_delta_lt"]
    )
    hierarchical = _number(record, "hierarchical_ci95_high") < high
    sequence = _number(record, "sequence_ci95_high") < high
    return point and hierarchical and sequence


def stage_decision(
    stage: str, evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose only the prespecified primary branch, after all diagnostics exist."""
    if stage not in ("67", "68", "69"):
        raise DecisionBlocked(f"unsupported fitted stage: {stage}")
    _ready(evidence)
    spec = protocol["stages"][stage]
    utility_checks = {
        r: _utility(_comparison(evidence, r), rule) for r, rule in spec["utility"].items()
    }
    utility = all(utility_checks.values())
    mechanisms: dict[str, bool] = {}
    if stage == "68":
        mechanisms = {
            r: _mechanism(_comparison(evidence, r), rule, spec["motion_both_ci95_high_lt"])
            for r, rule in spec["motion"].items()
        }
        if utility and all(mechanisms.values()):
            action = "RUN_STAGE69"
        elif utility:
            action = "GEOMETRY_UTILITY_WITHOUT_MOTION_EVIDENCE"
        else:
            action = "GEOMETRY_ROUTING_NEGATIVE"
    else:
        mechanisms = {
            r: _mechanism(
                _comparison(evidence, r),
                {"point_delta_lt": spec["mechanism_point_delta_lt"]},
                spec["mechanism_both_ci95_high_lt"],
            )
            for r in spec["mechanism_controls"]
        }
        action = (
            f"REPLICATE_STAGE{stage}"
            if utility
            else "RUN_STAGE68"
            if stage == "67"
            else "GEOMETRY_NEURAL_INCREMENT_NEGATIVE"
        )
    return {
        "stage": stage,
        "candidate": spec["candidate"],
        "utility_passed": utility,
        "utility_checks": utility_checks,
        "mechanism_checks": mechanisms,
        "next_action": action,
        "historical_acceptance": "INTEGRITY_BLOCKED",
        "claim_ceiling": "development_only",
    }


def replication_decision(
    stage: str,
    new_seed_evidence: Mapping[str, Mapping[str, Any]],
    averaged_loss_evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Replicate optimization conditional on frozen experts; no prediction ensemble."""
    if stage not in ("67", "69"):
        raise DecisionBlocked("this protocol replicates only neural selector stages")
    if set(new_seed_evidence) != {"13", "23"}:
        raise DecisionBlocked("exactly both replication seed results are required")
    per_seed: dict[str, bool] = {}
    for seed, result in new_seed_evidence.items():
        _ready(result)
        checks = [
            _number(_comparison(result, r), "point_delta") < 0.0
            for r in protocol["stages"][stage]["utility"]
        ]
        per_seed[seed] = all(checks)
    # Also validates all planned control diagnostics for seed-averaged losses.
    averaged = stage_decision(stage, averaged_loss_evidence, protocol)
    passed = all(per_seed.values()) and averaged["utility_passed"]
    return {
        "stage": stage,
        "new_seed_negative_deltas": per_seed,
        "seed_averaged_loss_utility_passed": averaged["utility_passed"],
        "next_action": "DEVELOPMENT_CANDIDATE_CONDITIONALLY_REPLICATED"
        if passed
        else "SELECTOR_REPLICATION_NEGATIVE",
        "fallback_authorized": False,
        "deployment_seed": 7,
        "claim_ceiling": "development_only",
    }
