from __future__ import annotations

from e_jepa_ttc.evaluation.collision_clock_gates import (
    evaluate_x0_height_gate,
    validate_x0_claim_scope,
)


def _passing_evidence() -> dict[str, float | int | bool]:
    return {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "coverage_drop_pp": 0.0,
        "delta_mid_vs_official_a5_oof": -3.1,
        "probability_delta_below_zero": 0.91,
        "paired_ci95_upper": -0.01,
        "identity_hashes_exact": True,
        "height_interface_bypassed": True,
        "global_transport_foreground_free": True,
        "motion_feature_schema_exact": True,
        "prefix_causality_passed": True,
        "forbidden_feature_audit_passed": True,
        "reference_identity": {
            "reference_family": "official_a5_oof",
            "path": "artifacts/a5.csv",
            "file_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
        },
    }


def test_height_gate_fails_closed_and_passes_only_complete_evidence() -> None:
    passing = _passing_evidence()
    assert evaluate_x0_height_gate(passing)["decision"] == "DYNAMIC_HEIGHT_BYPASS_SUPPORTED"
    incomplete = dict(passing)
    del incomplete["paired_ci95_upper"]
    assert evaluate_x0_height_gate(incomplete)["decision"] == "INCOMPLETE"
    negative = dict(passing)
    negative["delta_mid_vs_official_a5_oof"] = -2.9
    assert evaluate_x0_height_gate(negative)["decision"] == "NEGATIVE_DIRECT_PHASE"


def test_gate_rejects_nested_a5_under_official_name() -> None:
    evidence = _passing_evidence()
    evidence["reference_identity"] = {
        "reference_family": "nested_router_retrained_a5_constituent",
        "path": "artifacts/router.csv",
        "file_sha256": "c" * 64,
        "artifact_sha256": "d" * 64,
    }
    try:
        evaluate_x0_height_gate(evidence)
    except ValueError:
        pass
    else:
        raise AssertionError("nested A5 was accepted as the official gate reference")


def test_smoke_cannot_be_promoted_or_claim_geometry_free() -> None:
    payload = {
        "upstream_roi_is_box_conditioned": True,
        "explicit_foreground_height_interface_bypassed": True,
        "evidence_class": "smoke",
        "scientific_result": False,
    }
    validate_x0_claim_scope(payload, arm_id="X0-DYN-U")
    payload["geometry_free"] = True
    try:
        validate_x0_claim_scope(payload, arm_id="X0-DYN-U")
    except ValueError:
        pass
    else:
        raise AssertionError("geometry-free claim was accepted")
