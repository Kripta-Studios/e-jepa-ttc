"""Reference cases ported to production imports; lifecycle covered independently."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.risk_geometry_decision_v10 import (
    DecisionBlocked,
    replication_decision,
    stage_decision,
)

P = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "configs/protocol/scientific_recovery_v9_stage66_69.json"
    ).read_text()
)


def evidence(stage="67", delta=-3.0):
    spec = P["stages"][stage]
    refs = (
        set(spec["utility"]) | set(spec.get("mechanism_controls", [])) | set(spec.get("motion", {}))
    )
    return dict(
        execution_complete=True,
        protocol_conformant=True,
        diagnostics_complete=True,
        coverage_unchanged=True,
        finite_fraction=1.0,
        failure_count=0,
        comparisons={
            r: dict(
                point_delta=delta,
                hierarchical_ci95_high=-0.2,
                sequence_ci95_high=-0.1,
                weighted_sign_error_delta=0.0,
                crucial_bucket_mid_delta=0.0,
            )
            for r in refs
        },
    )


def test_stage67_negative_activates_geometry():
    assert stage_decision("67", evidence(delta=-1.0), P)["next_action"] == "RUN_STAGE68"


def test_utility_does_not_claim_structure():
    e = evidence()
    e["comparisons"]["S67-UNCONSTRAINED17"]["point_delta"] = 1.0
    r = stage_decision("67", e, P)
    assert r["next_action"] == "REPLICATE_STAGE67"
    assert not r["mechanism_checks"]["S67-UNCONSTRAINED17"]


@pytest.mark.parametrize(
    "key",
    ["execution_complete", "protocol_conformant", "diagnostics_complete", "coverage_unchanged"],
)
def test_missing_readiness_blocks_not_fallback(key):
    e = evidence()
    e[key] = False
    with pytest.raises(DecisionBlocked):
        stage_decision("67", e, P)


def test_missing_diagnostic_even_after_utility_fails():
    e = evidence(delta=3.0)
    del e["comparisons"]["S67-SIMPLEX8GATE"]
    with pytest.raises(DecisionBlocked):
        stage_decision("67", e, P)


def test_nan_cannot_pass():
    e = evidence()
    e["comparisons"]["S65-RISK17"]["point_delta"] = float("nan")
    with pytest.raises(DecisionBlocked):
        stage_decision("67", e, P)


def test_zero_ci_is_not_favorable():
    e = evidence()
    e["comparisons"]["S65-RISK17"]["sequence_ci95_high"] = 0.0
    assert stage_decision("67", e, P)["next_action"] == "RUN_STAGE68"


def test_sign_guardrail_blocks_utility():
    e = evidence()
    e["comparisons"]["S65-RISK17"]["weighted_sign_error_delta"] = 0.006
    assert not stage_decision("67", e, P)["utility_passed"]


def test_geometry_needs_motion_and_utility():
    e = evidence("68")
    assert stage_decision("68", e, P)["next_action"] == "RUN_STAGE69"
    e["comparisons"]["S68-GQUALITY45"]["point_delta"] = -0.5
    assert stage_decision("68", e, P)["next_action"] == "GEOMETRY_UTILITY_WITHOUT_MOTION_EVIDENCE"


def test_stage69_compares_both_baselines():
    e = evidence("69")
    e["comparisons"]["S68-GTRUE45"]["point_delta"] = -0.5
    assert stage_decision("69", e, P)["next_action"] == "GEOMETRY_NEURAL_INCREMENT_NEGATIVE"


def test_replication_failure_does_not_open_rescue():
    seeds = {"13": evidence(), "23": evidence(delta=0.1)}
    r = replication_decision("67", seeds, evidence(), P)
    assert r["next_action"] == "SELECTOR_REPLICATION_NEGATIVE" and not r["fallback_authorized"]


def test_replication_success():
    r = replication_decision("67", {"13": evidence(), "23": evidence()}, evidence(), P)
    assert (
        r["next_action"] == "DEVELOPMENT_CANDIDATE_CONDITIONALLY_REPLICATED"
        and r["deployment_seed"] == 7
    )


def test_missing_replication_seed_blocks():
    with pytest.raises(DecisionBlocked):
        replication_decision("67", {"13": evidence()}, evidence(), P)


def test_protocol_parameter_and_budget_consistency():
    assert len(P["data"]["feature17_order"]) == 17 and len(P["data"]["geometry28_order"]) == 28
    assert (
        P["stages"]["67"]["seed7_updates"]
        == len(P["stages"]["67"]["arms"]) * 3 * P["training"]["updates"]
    )
    assert P["max_branch_budget"]["negative_then_geometry_neural_updates"] == 18000 + 22500 + 45000
    assert P["resources"]["hour_limit"] is None
    assert P["historical_acceptance"] == "INTEGRITY_BLOCKED"
