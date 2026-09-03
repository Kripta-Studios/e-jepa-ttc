"""Machine-readable scope and eligibility gates for E-Clock X0."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REQUIRED_PROVENANCE = {
    "upstream_roi_is_box_conditioned": True,
    "explicit_foreground_height_interface_bypassed": True,
}


def validate_x0_claim_scope(payload: Mapping[str, Any], *, arm_id: str) -> None:
    """Reject provenance inflation and scientific claims from smoke artifacts."""

    if payload.get("upstream_roi_is_box_conditioned") is not True:
        raise ValueError("upstream_roi_is_box_conditioned=true is mandatory")
    bypass = payload.get("explicit_foreground_height_interface_bypassed")
    if arm_id in {"X0-BASE-U", "X0-DYN-U", "X0-DYN-W"} and bypass is not True:
        raise ValueError("height-bypass arms must declare the explicit interface bypass")
    if arm_id in {"X0-PAIR-U", "X0-A5-REPLAY"} and bypass is not False:
        raise ValueError("PAIR/A5 are geometry-infused and cannot claim height bypass")
    forbidden_true = (
        "geometry_free",
        "bbox_free",
        "detector_free",
        "absence_of_implicit_geometry",
        "perfect_physical_causality",
        "jepa_benefit",
        "garl_superiority",
        "state_of_the_art",
        "external_generalization",
    )
    if any(payload.get(field) is True for field in forbidden_true):
        raise ValueError("claim exceeds the authorized X0 scope")
    if payload.get("evidence_class") == "smoke" and payload.get("scientific_result") is not False:
        raise ValueError("smoke output cannot be a scientific result")


def evaluate_x0_height_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered DYN height-bypass gate without inferring missing evidence."""

    reference_identity = evidence.get("reference_identity")
    if not isinstance(reference_identity, Mapping):
        raise ValueError("gate requires a physical reference identity")
    required_reference = {"reference_family", "path", "file_sha256", "artifact_sha256"}
    if set(reference_identity) != required_reference:
        raise ValueError("gate reference identity is incomplete or ambiguous")
    if reference_identity.get("reference_family") != "official_a5_oof":
        raise ValueError("height gate requires official_a5_oof, never nested A5")
    if any(
        not isinstance(reference_identity[key], str) or not reference_identity[key]
        for key in required_reference
    ):
        raise ValueError("gate reference identity contains an empty field")
    numeric_rules = {
        "row_count": (lambda value: value == 8192),
        "finite_fraction": (lambda value: value == 1.0),
        "failure_rate": (lambda value: value == 0.0),
        "coverage_drop_pp": (lambda value: value <= 1.0),
        "delta_mid_vs_official_a5_oof": (lambda value: value <= -3.0),
        "probability_delta_below_zero": (lambda value: value >= 0.90),
        "paired_ci95_upper": (lambda value: value < 0.0),
    }
    boolean_fields = (
        "identity_hashes_exact",
        "height_interface_bypassed",
        "global_transport_foreground_free",
        "motion_feature_schema_exact",
        "prefix_causality_passed",
        "forbidden_feature_audit_passed",
    )
    missing = sorted((set(numeric_rules) | set(boolean_fields)) - set(evidence))
    if missing:
        return {"decision": "INCOMPLETE", "missing": missing, "checks": {}}
    checks: dict[str, bool] = {}
    for field, rule in numeric_rules.items():
        value = evidence[field]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {"decision": "INCOMPLETE", "non_finite": field, "checks": checks}
        checks[field] = rule(float(value))
    checks.update({field: evidence[field] is True for field in boolean_fields})
    decision = (
        "DYNAMIC_HEIGHT_BYPASS_SUPPORTED" if all(checks.values()) else "NEGATIVE_DIRECT_PHASE"
    )
    return {"decision": decision, "missing": [], "checks": checks}


def evaluate_x0_primary_dyn_vs_base_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen primary DYN-vs-BASE gate without an invented effect threshold."""

    numeric_rules = {
        "row_count": lambda value: value == 8192,
        "base_finite_fraction": lambda value: value == 1.0,
        "dyn_finite_fraction": lambda value: value == 1.0,
        "base_failure_rate": lambda value: value == 0.0,
        "dyn_failure_rate": lambda value: value == 0.0,
        "coverage_delta_pp": lambda value: value == 0.0,
        "finite_draw_fraction": lambda value: 0.0 < value <= 1.0,
        "paired_ci95_upper": lambda value: value < 0.0,
    }
    boolean_fields = (
        "primary_comparison_signed",
        "identity_hashes_exact",
        "matched_config_contract",
        "paired_identical_draws",
        "incomplete_draws_disclosed",
        "official_a5_not_used_as_primary_reference",
    )
    missing = sorted((set(numeric_rules) | set(boolean_fields)) - set(evidence))
    if missing:
        return {"decision": "INCOMPLETE", "passed": False, "missing": missing, "checks": {}}
    checks: dict[str, bool] = {}
    for field, rule in numeric_rules.items():
        value = evidence[field]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {
                "decision": "INCOMPLETE",
                "passed": False,
                "missing": [],
                "non_finite": field,
                "checks": checks,
            }
        checks[field] = rule(float(value))
    checks.update({field: evidence[field] is True for field in boolean_fields})
    passed = all(checks.values())
    return {
        "decision": ("DYNAMIC_SLOTS_SUPPORTED" if passed else "DYNAMIC_SLOTS_NOT_SUPPORTED"),
        "passed": passed,
        "missing": [],
        "checks": checks,
        "effect_size_threshold_mid": None,
        "effect_rule": "paired_ci95_upper_strictly_below_zero",
    }


__all__ = [
    "REQUIRED_PROVENANCE",
    "evaluate_x0_height_gate",
    "evaluate_x0_primary_dyn_vs_base_gate",
    "validate_x0_claim_scope",
]
