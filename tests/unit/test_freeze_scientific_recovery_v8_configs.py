"""Regression tests for the deterministic, train-only V8 protocol freeze."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (
    V8IntegrityError,
    verify_frozen_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/freeze_scientific_recovery_v8_configs.py"
SPEC = importlib.util.spec_from_file_location("freeze_v8", SCRIPT)
assert SPEC and SPEC.loader
FREEZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FREEZE
SPEC.loader.exec_module(FREEZE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze() -> dict[str, object]:
    assert FREEZE.main(["--protocol", "configs/protocol/scientific_recovery_v8_temporal.json"]) == 0
    return json.loads(
        (
            ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json"
        ).read_text()
    )


def test_freeze_is_idempotent_and_signatures_are_valid() -> None:
    first = _freeze()
    files = [ROOT / "configs/protocol/scientific_recovery_v8_temporal.json"]
    files.extend(ROOT / item["path"] for item in first["enabled_seed7_configs"].values())
    files.extend(ROOT / item["path"] for item in first["model_configs"].values())
    files.extend(ROOT / item["path"] for item in first["c1_analysis_plans"].values())
    files.append(ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json")
    before = {file: _sha(file) for file in files}
    second = _freeze()
    assert before == {file: _sha(file) for file in files}
    assert FREEZE.artifact_hash(first) == first["artifact_sha256"]
    assert FREEZE.artifact_hash(second) == second["artifact_sha256"]
    protocol = json.loads(files[0].read_text())
    assert FREEZE.artifact_hash(protocol) == protocol["artifact_sha256"]


def test_freeze_verify_checks_existing_signed_outputs() -> None:
    _freeze()
    assert (
        FREEZE.main(
            ["--verify", "--protocol", "configs/protocol/scientific_recovery_v8_temporal.json"]
        )
        == 0
    )


def test_freeze_verify_rejects_tampered_c1_analysis_plan() -> None:
    manifest = _freeze()
    protocol_path = ROOT / "configs/protocol/scientific_recovery_v8_temporal.json"
    plan_path = ROOT / manifest["c1_analysis_plans"]["router_regime"]["path"]
    original = plan_path.read_bytes()
    try:
        plan_path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="C1 analysis plan"):
            FREEZE.verify_frozen_outputs(protocol_path)
    finally:
        plan_path.write_bytes(original)
        _freeze()


def test_runner_rejects_tampered_model_recipe_and_missing_integrity_hashes() -> None:
    manifest = _freeze()
    protocol_path = ROOT / "configs/protocol/scientific_recovery_v8_temporal.json"
    manifest_path = (
        ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json"
    )
    model_path = ROOT / manifest["model_configs"]["timevol20_3"]["path"]
    plan_path = ROOT / manifest["c1_analysis_plans"]["router_regime"]["path"]
    original_model = model_path.read_bytes()
    original_plan = plan_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    try:
        model_path.write_bytes(original_model + b"# tampered\n")
        with pytest.raises(V8IntegrityError, match="model configuration hash mismatch"):
            verify_frozen_inputs(protocol_path, manifest_path)
        model_path.write_bytes(original_model)
        plan_path.write_bytes(original_plan + b"\n")
        with pytest.raises(V8IntegrityError, match="C1 analysis plan hash mismatch"):
            verify_frozen_inputs(protocol_path, manifest_path)
        plan_path.write_bytes(original_plan)
        tampered_manifest = json.loads(original_manifest)
        tampered_manifest["integrity"].pop("mid_sample_weight_sha256")
        sign_artifact(tampered_manifest)
        manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
        with pytest.raises(V8IntegrityError, match="mid_sample_weight_sha256"):
            verify_frozen_inputs(protocol_path, manifest_path)
    finally:
        model_path.write_bytes(original_model)
        plan_path.write_bytes(original_plan)
        manifest_path.write_bytes(original_manifest)
        _freeze()


def test_row_target_mismatch_fails_closed(tmp_path: Path) -> None:
    header = "sample_token,sequence_id,track_id,target_ttc_s,fold\n"
    a5 = tmp_path / "a5.csv"
    garl = tmp_path / "garl.csv"
    a5.write_text(header + "token,sequence,track,1.0,0\n")
    garl.write_text(header + "token,sequence,track,1.1,0\n")
    with pytest.raises(ValueError, match="target_ttc_s mismatch"):
        FREEZE.derive_samples(a5, garl)


def test_sealed_split_rejected() -> None:
    with pytest.raises(ValueError, match="sealed split"):
        FREEZE.reject_sealed_paths({"path": "artifacts/public_validation/predictions.csv"})


def test_checkpoint_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    baselines = FREEZE.read_signed(ROOT / FREEZE.V7_BASELINES)
    monkeypatch.setattr(FREEZE, "sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="checkpoint hash mismatch for A5 fold 0"):
        FREEZE.checkpoints(baselines)


def test_router_inner_folds_are_disjoint_and_features_are_fixed() -> None:
    manifest = _freeze()
    for fold in range(3):
        config = yaml.safe_load(
            (
                ROOT
                / "configs/experiment/scientific_recovery_v8_fold_chain"
                / f"router_fold{fold}_seed7.yaml"
            ).read_text()
        )
        outer_dev = set(config["outer_dev_sequence_ids"])
        all_inner_dev: set[str] = set()
        for inner in config["inner_folds"]:
            assert not (set(inner["train_sequence_ids"]) & set(inner["dev_sequence_ids"]))
            assert not (set(inner["dev_sequence_ids"]) & outer_dev)
            all_inner_dev.update(inner["dev_sequence_ids"])
        assert all_inner_dev == set(config["outer_train_sequence_ids"])
        assert config["router"]["feature_names"] == FREEZE.ROUTER_FEATURES
        assert not set(config["router"]["feature_names"]) & set(
            config["router"]["prohibited_features"]
        )
        assert config["router"]["pipeline"]["class_weight"] is None
        assert config["router"]["fit_sample_weight_only"] is True
        assert config["router"]["label"] == "lower_raw_official_eta_mid_loss"
        assert config["router"]["effective_sample_weight"] == (
            "sequence_macro_mid_row_weight_times_absolute_expert_loss_delta"
        )
    assert len(manifest["enabled_seed7_configs"]) == 9


def test_b_configs_preserve_a5_contract_except_frontend_cache_and_channels() -> None:
    b1 = yaml.safe_load(
        (
            ROOT
            / "configs/experiment/scientific_recovery_v8_fold_chain/timevol20_3_fold0_seed7.yaml"
        ).read_text()
    )
    b2 = yaml.safe_load(
        (
            ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/exp6_3_fold0_seed7.yaml"
        ).read_text()
    )
    for section in ("training", "loss", "decision_contract"):
        assert b1[section] == b2[section]
    assert b1["data"]["roi_size"] == b2["data"]["roi_size"] == [128, 128]
    assert b1["data"]["steps"] == b2["data"]["steps"] == 3
    assert b1["data"]["channels_per_endpoint"] == 20
    assert b2["data"]["channels_per_endpoint"] == 6
    teacher = b1["data"]["dinov3_relational_teacher"]
    assert teacher == b2["data"]["dinov3_relational_teacher"]
    assert teacher["manifest"].endswith(
        "dinov3_convnext_large_relational_a4_train8192_rgb_v1/manifest.json"
    )
    assert teacher["artifact_sha256"] == (
        "6511c684881f3360efb7c8718976ef6b37de36668a89d9ea2f8d4bdf6f620b20"
    )
    assert (
        b1["training"]["representation_teacher_cache_artifact_sha256"] == teacher["artifact_sha256"]
    )


def test_conditional_templates_are_disabled_and_models_have_declared_channels() -> None:
    manifest = _freeze()
    templates = manifest["conditional_templates"]
    assert templates["pair20_2"]["enabled"] is False
    assert templates["gated_exp6_3"]["enabled"] is False
    assert "signed" in templates["pair20_2"]["runner_refusal"]
    pair = yaml.safe_load((ROOT / FREEZE.MODEL_PATHS["pair20_2"]).read_text())
    gated = yaml.safe_load((ROOT / FREEZE.MODEL_PATHS["gated_exp6_3"]).read_text())
    assert pair["in_channels"] == 20
    assert gated["in_channels"] == 6


def test_protocol_freezes_v8_specific_gates_and_auditable_frontend_capacity() -> None:
    manifest = _freeze()
    protocol = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v8_temporal.json").read_text()
    )
    assert protocol["v8_supersession"]["provisional_post_v7_commit"] == (
        "f9331b29596c4107430af5a8c78935bd127ccf94"
    )
    assert protocol["v8_supersession"]["historical_v7_a4_retention_gate"] == "immutable"
    assert "a4_retention" not in protocol["gates"]["ttc_candidate_gate"]
    assert protocol["gates"]["mechanistic_interpretability_gate"]["a4_retention"] == "diagnostic"
    assert protocol["training_contract"]["multiseed_replication_seeds"] == [13, 23]
    assert protocol["external_confirmation"]["requires_user_authorization"] is True
    assert protocol["router_contract"]["classifier"]["class_weight"] is None
    assert protocol["router_contract"]["fit_parameter"] == "router__sample_weight"
    capacity = protocol["timevol20_3_capacity"]
    assert capacity["a5"]["input_stem_params"] == 9600
    assert capacity["timevol20_3"]["input_stem_params"] == 16000
    assert capacity["delta_input_stem_params"] == 6400
    assert capacity["delta_total_params"] == 6400
    assert capacity["a5"]["input_shape"] == [1, 3, 12, 128, 128]
    assert capacity["timevol20_3"]["input_shape"] == [1, 3, 20, 128, 128]
    exp6 = protocol["exp6_source_parity"]
    assert exp6["official_commit_sha"] == "59c498b71ae526bc2d7e570c82a078306a996b93"
    assert exp6["time_bins"] == 35
    assert protocol["jepa_d4_contract"]["cross_track_derangement"] is True
    assert (
        protocol["sample_contract"]["target_identity_sha256"]
        == protocol["sample_contract"]["target_sha256"]
    )
    counts = protocol["sample_contract"]["row_count_contract"]
    assert counts["total"] == 8192
    assert sum(counts["by_outer_fold"].values()) == 8192
    assert sum(counts["by_sequence"].values()) == 8192
    assert sum(counts["by_bucket"].values()) == 8192
    plans = protocol["c1_analysis_plans"]
    assert plans == manifest["c1_analysis_plans"]
    assert set(plans) == {"autopsy_h3", "exp6_regime", "router_regime"}
    assert all("artifact_sha256" in plan for plan in plans.values())
    router_plan = json.loads((ROOT / plans["router_regime"]["path"]).read_text())
    autopsy_plan = json.loads((ROOT / plans["autopsy_h3"]["path"]).read_text())
    assert router_plan["source_aggregate_contract"]["primary_ttc_gate_required"] is True
    assert (
        router_plan["source_aggregate_contract"]["aggregate_schema"]
        == protocol["evaluation_contract"]["aggregate_schema"]
    )
    assert set(router_plan["source_aggregate_contract"]["config_sha256_by_fold"]) == {
        "0",
        "1",
        "2",
    }
    assert set(autopsy_plan["source_aggregate_contract"]["required_outputs"]) == {
        "factorial_replay",
        "diagnostic",
    }
    assert (
        len(autopsy_plan["source_aggregate_contract"]["factorial_replay_schema"]["combinations"])
        == 5
    )
    assert (
        autopsy_plan["source_aggregate_contract"]["diagnostic_schema"]["decision_rule"]["otherwise"]
        == "H2"
    )


def test_v8_dino_teacher_contract_is_grouped_8192_not_screen_2048() -> None:
    teacher, artifact = FREEZE.a5_dino_teacher_contract()
    assert teacher["manifest"] == (
        "artifacts/cache/dinov3_convnext_large_relational_a4_train8192_rgb_v1/manifest.json"
    )
    assert artifact == "6511c684881f3360efb7c8718976ef6b37de36668a89d9ea2f8d4bdf6f620b20"
    assert teacher["source_config"].endswith("a5_causal_fold0.yaml")
    manifest = json.loads((ROOT / teacher["manifest"]).read_text(encoding="utf-8"))
    assert int(manifest["scope"]["row_count"]) == 8192
