"""A1 evaluation must retain both complete endpoints and the prospective admission gates."""

from copy import deepcopy

import pytest

from operational.joint_measurement import adaptation_evaluate as evaluation
from operational.joint_measurement import adaptation_inference as inference
from operational.joint_measurement import evaluate as primary
from operational.range_transfer.common import digest, read, sha, write


@pytest.fixture
def pair(tmp_path, monkeypatch):
    registration = tmp_path / "REGISTRATION.json"
    write(registration, {"test": "A1"})
    monkeypatch.setattr(evaluation.a1, "DIRECTORY", tmp_path)
    monkeypatch.setattr(evaluation.a1, "REGISTRATION", registration)
    monkeypatch.setattr(evaluation.a1, "registered", lambda: {})
    monkeypatch.setattr(evaluation, "DIRECTORY", tmp_path)
    monkeypatch.setattr(evaluation, "EVALUATION", tmp_path / "evaluation/REGISTRATION.json")
    monkeypatch.setattr(
        evaluation.reused, "freeze_primary", lambda: {"endpoints": {evaluation.BASELINE: "a5"}}
    )
    endpoints = {}
    for mode in evaluation.a1.MODES:
        target = evaluation.a1.folder(mode)
        target.mkdir(parents=True)
        payload = mode.encode()
        (target / "checkpoint_last.pt").write_bytes(payload)
        versions = target / "checkpoint_versions"
        versions.mkdir()
        (versions / "v.pt").write_bytes(payload)
        checkpoint_hash = sha(target / "checkpoint_last.pt")
        identity = {"registration_sha256": sha(registration), "mode": mode}
        identity["identity_sha256"] = digest(identity)
        write(target / "IDENTITY.json", identity)
        write(
            target / "CHECKPOINT_RECEIPT.json",
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 12000,
                "identity_sha256": identity["identity_sha256"],
                "checkpoint_sha256": checkpoint_hash,
            },
        )
        write(
            target / "UPDATE_JOURNAL.json",
            {
                "completed_updates": 12000,
                "durable_updates": 12000,
                "pending_update_upper": 0,
                "identity_sha256": identity["identity_sha256"],
            },
        )
        write(
            target / "CHECKPOINT_POINTER.json",
            {
                "completed_updates": 12000,
                "checkpoint_sha256": checkpoint_hash,
                "version": "v.pt",
            },
        )
        endpoints[target.name] = checkpoint_hash
    write(tmp_path / "ENDPOINTS.json", endpoints)
    return tmp_path


def test_both_endpoints_required_and_reused_context_restored(pair):
    before = evaluation.reused.DIRECTORY
    result = evaluation.freeze_endpoints()
    assert set(result["endpoints"]) == {evaluation.FIT, evaluation.CONTROL, evaluation.BASELINE}
    assert evaluation.reused.DIRECTORY == before
    endpoints = read(pair / "ENDPOINTS.json")
    del endpoints[evaluation.CONTROL]
    write(pair / "ENDPOINTS.json", endpoints)
    with pytest.raises(ValueError, match="every stage"):
        evaluation.freeze_endpoints()
    assert evaluation.reused.DIRECTORY == before


@pytest.mark.parametrize("fault", ["durability", "pending", "pointer", "weights", "identity"])
def test_no_partial_or_altered_control_can_enter_evaluation(pair, fault):
    target = evaluation.a1.folder("FULL")
    if fault in ("durability", "pending"):
        path = target / "UPDATE_JOURNAL.json"
        record = read(path)
        record["durable_updates" if fault == "durability" else "pending_update_upper"] = 1
        write(path, record)
    elif fault == "pointer":
        path = target / "CHECKPOINT_POINTER.json"
        record = read(path)
        record["completed_updates"] = 11999
        write(path, record)
    elif fault == "weights":
        (target / "checkpoint_versions/v.pt").write_bytes(b"altered")
    else:
        path = target / "IDENTITY.json"
        record = read(path)
        record["mode"] = "LAST-BLOCK"
        write(path, record)
    with pytest.raises(ValueError, match="endpoint|identity"):
        evaluation.freeze_endpoints()


@pytest.mark.parametrize("name,role", [(evaluation.FIT, "V"), ("UNKNOWN", "H")])
def test_unregistered_population_or_endpoint_rejected_before_io(name, role):
    with pytest.raises(ValueError, match="Only the two"):
        inference.predict(name, role)


def metrics(overall=100.0, crucial=100.0, negative=100.0, sign=0.05):
    return {
        "overall_MiD": overall,
        "bins": {
            name: {"group_macro_mid": value}
            for name, value in {
                "crucial": crucial,
                "small": 100.0,
                "large": 100.0,
                "negative": negative,
            }.items()
        },
        "diagnostics": {"sign_error_rate": sign},
        "complete_prediction_and_mid_coverage": True,
    }


@pytest.fixture
def gates(monkeypatch):
    selection = {
        "all_routes_sign_increase_max": 0.005,
        "all_routes_band_ratio_max": 1.1,
        "max_integrations": 2,
        "routes": {
            "overall_ratio_max": 0.97,
            "crucial_ratio_max": 0.95,
            "crucial_route_overall_ratio_max": 1.02,
            "cost_route_overall_ratio_max": 1.02,
            "cost_ratio_max": 0.7,
        },
    }
    monkeypatch.setattr(primary, "config", lambda: {"selection": selection})
    return {"development_gates": selection}


def choose(fit, control, gates, cost=0.6):
    cohorts = {
        f"H/{evaluation.FIT}": fit,
        f"H/{evaluation.CONTROL}": control,
        f"H/{evaluation.BASELINE}": metrics(),
    }
    timing = {
        "results": {
            name: {"p50_seconds": 1.0 if name == evaluation.BASELINE else cost}
            for name in (evaluation.FIT, evaluation.CONTROL, evaluation.BASELINE)
        }
    }
    return evaluation.select(cohorts, timing, gates)


def test_mechanism_signal_cannot_bypass_absolute_negative_band_gate(gates):
    decision = choose(
        metrics(overall=80, crucial=80, negative=120),
        metrics(overall=90, crucial=90, negative=130),
        gates,
    )
    assert decision["paired_mechanism"]["eligible"]
    assert decision["status"] == "NO_CANDIDATE_A1"
    assert not decision["system_promotion"]
    assert not decision["decisions"][evaluation.FIT]["cost_route_pass"]


@pytest.mark.parametrize("violation", ["sign", "coverage"])
def test_large_speed_gain_cannot_bypass_sign_or_coverage(gates, violation):
    fit = metrics(overall=80, crucial=80)
    if violation == "sign":
        fit["diagnostics"]["sign_error_rate"] = 0.06
    else:
        fit["complete_prediction_and_mid_coverage"] = False
    decision = choose(fit, deepcopy(fit), gates, cost=0.1)
    assert decision["ranked"] == []


def test_ranking_tie_uses_critical_metric_and_cost_only_keeps_constraints(gates):
    decision = choose(metrics(overall=95, crucial=90), metrics(overall=95, crucial=85), gates)
    assert decision["ranked"] == [evaluation.CONTROL, evaluation.FIT]
    cost_only = choose(metrics(), metrics(), gates)
    assert cost_only["decisions"][evaluation.FIT]["cost_route_pass"]
    assert not cost_only["system_promotion"]


def test_stale_inference_proof_cannot_be_reused_with_new_sources(pair, monkeypatch):
    proof_dir = pair / "evaluation/inference_runtime"
    write(evaluation.EVALUATION, {"test": "evaluation"})
    write(
        proof_dir / "REGISTRATION.json",
        {
            "evaluation_registration_sha256": sha(evaluation.EVALUATION),
            "endpoints": evaluation.freeze_endpoints(),
            "sources": {"old": "hash"},
        },
    )
    write(
        proof_dir / "RESULT.json",
        {"status": "ADMITTED", "registration_sha256": sha(proof_dir / "REGISTRATION.json")},
    )
    monkeypatch.setattr(evaluation, "source_binding", lambda: {"new": "hash"})
    with pytest.raises(ValueError, match="sources or endpoints"):
        inference.predict(evaluation.FIT, "P")


@pytest.mark.parametrize(
    "changed", [["encoder.0.weight"], ["encoder.0.weight", "encoder.7.weight"]]
)
def test_actual_weight_comparison_rejects_frozen_prefix_even_with_valid_receipts(
    pair, monkeypatch, changed
):
    monkeypatch.setattr(
        evaluation.a1,
        "registered",
        lambda: {
            "parameter_policy": {
                "LAST-BLOCK": {"frozen": ["encoder.0.weight"], "last_block_prefix": "encoder.7."}
            }
        },
    )
    monkeypatch.setattr(
        evaluation.reused,
        "verify_stage",
        lambda *args: {"encoder_tensors_changed": changed, "readout_tensors_changed": ["head"]},
    )
    with pytest.raises(ValueError, match="adaptation boundary"):
        evaluation.verify_endpoint("LAST-BLOCK")
