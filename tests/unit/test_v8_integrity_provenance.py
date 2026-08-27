from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from e_jepa_ttc.data.canonical_token_identity import (
    hash_canonical_json_records,
    hash_ordered_token_ids,
    hash_sorted_token_strings,
)
from e_jepa_ttc.data.scientific_recovery_v5 import _values_sha256
from e_jepa_ttc.data.scientific_recovery_v8_cache import _canonical_records, _protocol_rows, _token_hash
from e_jepa_ttc.evaluation.scientific_recovery_v8_aggregate import _records_hash
from e_jepa_ttc.models.causal_expert_router import _canonical_token_hash
from e_jepa_ttc.scientific_provenance import (
    AUTOPSY_REPLAY_IDENTITY_PATHS,
    FORBIDDEN_SCIENTIFIC_ENV,
    ScientificProvenanceError,
    assert_autopsy_replay_producer_reusable,
    assert_router_expert_reusable,
    observe_git_identity,
    refuse_scientific_bypass_env,
    require_clean_scientific_worktree,
    serialize_git_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = _load_script(
    "a5_train_screen_contract", ROOT / "scripts" / "train_causal_scale_eap_screen.py"
)
ROUTER_EXPERT = _load_script(
    "v8_router_expert", ROOT / "scripts" / "train_scientific_recovery_v8_router_expert.py"
)
FREEZE = _load_script("v8_freeze", ROOT / "scripts" / "freeze_scientific_recovery_v8_configs.py")
ROUTER_AGG = _load_script(
    "v8_router_aggregate", ROOT / "scripts" / "aggregate_scientific_recovery_v8_router.py"
)


def test_missing_preflight_cannot_be_synthesized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing_preflight.json"
    monkeypatch.setattr(SCREEN, "ROOT", tmp_path)
    contract = {
        "representation_change": {
            "type": "a4_endpoint_dino_plus_event_native_local_cross_time_transport",
            "dino_endpoint_teacher_unchanged_from_a4": True,
            "dino_temporal_delta_removed": True,
            "transport_model_input": "event_dense_features_only",
            "bbox_used_by_transport": False,
            "rgb_used_at_inference": False,
            "jepa_objective": False,
            "direct_ttc_regressor_from_flow": False,
            "analytic_height_ratio_remains_primary_backbone": True,
            "transport_radius": 1,
            "transport_pairs": ["t0_to_t1", "t1_to_t2"],
        },
        "preflight_contract": {
            "artifact_type": "a5_transport_preflight_train_only_v3_confirmation",
            "artifact": missing.as_posix(),
            "artifact_sha256": "a" * 64,
            "file_sha256": "b" * 64,
            "selected_radius": 1,
            "selected_temperature": 0.02,
        },
    }

    def _resolve(value: object) -> Path:
        if not isinstance(value, str):
            raise ValueError("path references must be strings")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(SCREEN, "_resolve", _resolve)
    training = type(
        "T",
        (),
        {
            "representation_supervision": "dinov3_local_relational",
            "representation_temporal_delta_weight": 0.0,
            "initialization_mode": "none",
            "freeze_encoder": False,
        },
    )()
    model = type(
        "M",
        (),
        {"transport_enabled": True, "transport_radius": 1, "transport_temperature": 0.02},
    )()
    with pytest.raises(FileNotFoundError):
        SCREEN._validate_a5_transport_change(training, model, contract)


def test_a5_preflight_source_does_not_synthesize_missing_artifacts() -> None:
    source = (ROOT / "scripts" / "train_causal_scale_eap_screen.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_a5_transport_change"
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if getattr(node.type, "id", None) == "FileNotFoundError":
            raise AssertionError("A5 preflight must not catch FileNotFoundError")


def test_observed_hash_cannot_fall_back_to_expected_hash() -> None:
    source = (ROOT / "scripts" / "train_causal_scale_eap_screen.py").read_text(encoding="utf-8")
    assert "observed_file_sha = preflight.get(" not in source
    assert "observed_hash = expected_hash" not in source


def test_cross_component_canonical_token_hashes_match() -> None:
    tokens = ("zeta-token", "alpha-token", "mid-token")
    expected_sorted = hash_sorted_token_strings(tokens)
    assert SCREEN._sorted_values_sha256(list(tokens)) == expected_sorted
    assert _values_sha256(tokens) == expected_sorted
    train_hash = hash_sorted_token_strings(["zeta-token", "alpha-token"])
    newline_join = hashlib.sha256(
        "\n".join(sorted(["zeta-token", "alpha-token"])).encode()
    ).hexdigest()
    assert train_hash != newline_join
    assert SCREEN._sorted_values_sha256(["zeta-token", "alpha-token"]) == train_hash
    expert_source = (ROOT / "scripts" / "train_scientific_recovery_v8_router_expert.py").read_text(
        encoding="utf-8"
    )
    assert "hash_sorted_token_strings" in expert_source
    assert '"\\n".join(sorted(' not in expert_source
    nested_source = (ROOT / "scripts" / "run_scientific_recovery_v8_nested_router.py").read_text(
        encoding="utf-8"
    )
    assert "hash_sorted_token_strings" in nested_source
    assert '"\\n".join(sorted(' not in nested_source
    assert "bind_expert_oof_to_trainer_point_ttc" in nested_source
    assert "routing_point_ttc" in expert_source
    assert "point_prediction_ttc_s" in expert_source
    assert '"prediction_ttc": source.prediction_ttc_s' not in expert_source
    assert '"finite": True' not in expert_source
    all_trainings = (ROOT / "scripts" / "run_scientific_recovery_v8_all_trainings.ps1").read_text(
        encoding="utf-8"
    )
    assert "--verify" in all_trainings
    assert "00_verify_freeze" in all_trainings
    assert "00_freeze" not in all_trainings
    freeze_calls = [
        line
        for line in all_trainings.splitlines()
        if "freeze_scientific_recovery_v8_configs.py" in line
    ]
    assert freeze_calls
    assert all("--verify" in line for line in freeze_calls)
    assert "$null = $p.Handle" in all_trainings
    assert "$null = $proc.Handle" in all_trainings
    assert "exited without reporting an exit code" in all_trainings
    assert "WaitForExit($script:HeartbeatIntervalMs)" in all_trainings
    assert "LIVE" in all_trainings
    assert "Write-TrainerProgressSnapshot" in all_trainings
    assert "System.Collections.Generic.List[object]" in all_trainings
    assert "Get-JsonNote" in all_trainings
    assert "$files += @(" not in all_trainings
    assert _canonical_token_hash(("zeta-token", "alpha-token", "mid-token")) == expected_sorted
    assert _token_hash(tokens) == expected_sorted

    records = [{"token_id": token} for token in tokens]
    expected_records = hash_canonical_json_records(records)
    assert hash_ordered_token_ids(tokens) == expected_records
    assert FREEZE.canonical_records(list(records)) == expected_records
    assert _records_hash(records) == expected_records
    assert ROUTER_AGG._canonical_records(records) == expected_records
    assert _canonical_records(records) == expected_records


def test_v8_cache_protocol_identity_uses_ordered_token_ids() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "protocol" / "scientific_recovery_v8_temporal.json").read_text(
            encoding="utf-8"
        )
    )
    rows, token_hash, folds = _protocol_rows(protocol)
    contract = protocol["sample_contract"]
    assert rows == 8192
    assert token_hash == contract["ordered_token_ids_sha256"]
    assert "sorted_sample_tokens_sha256" not in contract
    assert isinstance(folds, list) and len(folds) == 3
    source = (ROOT / "src" / "e_jepa_ttc" / "data" / "scientific_recovery_v8_cache.py").read_text(
        encoding="utf-8"
    )
    assert 'contract.get("sorted_sample_tokens_sha256")' not in source
    assert "hash_ordered_token_ids(key[1] for key in selected_keys)" in source
    assert "require_protocol_identity=True requires --protocol" in source


def _init_temp_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("print(0)\n", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "V8",
        "GIT_AUTHOR_EMAIL": "v8@example.test",
        "GIT_COMMITTER_NAME": "V8",
        "GIT_COMMITTER_EMAIL": "v8@example.test",
    }
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=env,
    )


def test_dirty_scientific_worktree_fails(tmp_path: Path) -> None:
    _init_temp_repo(tmp_path)
    (tmp_path / "tracked.py").write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(ScientificProvenanceError, match="clean Git worktree"):
        require_clean_scientific_worktree(tmp_path)


def test_git_dirty_cannot_be_falsely_serialized_as_clean(tmp_path: Path) -> None:
    _init_temp_repo(tmp_path)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("print(1)\n", encoding="utf-8")
    observed = observe_git_identity(tmp_path)
    assert observed["git_dirty"] is True
    with pytest.raises(ScientificProvenanceError, match="git_dirty=false"):
        serialize_git_identity({**observed, "git_dirty": False})
    serialized = serialize_git_identity(observed)
    assert serialized["git_dirty"] is True


def test_invalid_router_experts_cannot_aggregate() -> None:
    with pytest.raises(ScientificProvenanceError, match="dirty worktree"):
        assert_router_expert_reusable({"status": "completed", "git_dirty": True, "fixture": False})
    with pytest.raises(ScientificProvenanceError, match="failed_integrity"):
        assert_router_expert_reusable({"status": "failed_integrity", "git_dirty": False})
    with pytest.raises(ScientificProvenanceError, match="fixture"):
        assert_router_expert_reusable({"status": "completed", "git_dirty": False, "fixture": True})
    with pytest.raises(ScientificProvenanceError, match="dirty worktree"):
        assert_router_expert_reusable({"status": "completed", "fixture": False})


def test_autopsy_replay_reuse_requires_clean_producer_head() -> None:
    complete = {
        "status": "completed_replay_without_optimizer_steps",
        "git_commit": "abc123",
        "git_dirty": False,
    }
    assert_autopsy_replay_producer_reusable(complete, expected_commit="abc123")
    with pytest.raises(ScientificProvenanceError, match="no git_commit"):
        assert_autopsy_replay_producer_reusable(
            {"status": "completed_replay_without_optimizer_steps"},
            expected_commit="abc123",
        )
    with pytest.raises(ScientificProvenanceError, match="clean worktree"):
        assert_autopsy_replay_producer_reusable(
            {**complete, "git_dirty": True},
            expected_commit="abc123",
        )
    with pytest.raises(ScientificProvenanceError, match="differs from implementation HEAD"):
        assert_autopsy_replay_producer_reusable(complete, expected_commit="def456")
    all_trainings = (ROOT / "scripts" / "run_scientific_recovery_v8_all_trainings.ps1").read_text(
        encoding="utf-8"
    )
    assert "assert_scientific_recovery_v8_autopsy_replay_reusable.py" in all_trainings
    assert "Test-Path -LiteralPath $A5ReplayManifest" not in all_trainings
    assert "Reusing completed A5/C2F/Garl autopsy replay manifests" not in all_trainings
    replay = (ROOT / "scripts" / "replay_scientific_recovery_v8_mechanisms.py").read_text(
        encoding="utf-8"
    )
    assert "_producer_git_identity" in replay
    aggregator = (ROOT / "scripts" / "aggregate_scientific_recovery_v8_autopsy.py").read_text(
        encoding="utf-8"
    )
    assert "assert_autopsy_replay_producer_reusable" in aggregator
    assert '"producer_git_commit_matches_head": True' not in aggregator
    assert "a5_source.get(\"git_commit\")" in aggregator
    assert "src/e_jepa_ttc/models/causal_scale_ttc.py" in AUTOPSY_REPLAY_IDENTITY_PATHS
    assert "scripts/run_scientific_recovery_v8_nested_router.py" not in AUTOPSY_REPLAY_IDENTITY_PATHS
    assert "src/e_jepa_ttc/scientific_provenance.py" not in AUTOPSY_REPLAY_IDENTITY_PATHS
    assert "src/e_jepa_ttc/data/scientific_recovery_v8_cache.py" not in AUTOPSY_REPLAY_IDENTITY_PATHS


def test_autopsy_replay_reuse_allows_unrelated_head_advance(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = {
        "status": "completed_replay_without_optimizer_steps",
        "git_commit": "aaa111",
        "git_dirty": False,
    }
    monkeypatch.setattr(
        "e_jepa_ttc.scientific_provenance.autopsy_replay_identity_paths_changed",
        lambda producer, head, root=None: False,
    )
    assert_autopsy_replay_producer_reusable(complete, expected_commit="bbb222")


def test_autopsy_replay_reuse_rejects_identity_file_change(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = {
        "status": "completed_replay_without_optimizer_steps",
        "git_commit": "aaa111",
        "git_dirty": False,
    }
    monkeypatch.setattr(
        "e_jepa_ttc.scientific_provenance.autopsy_replay_identity_paths_changed",
        lambda producer, head, root=None: True,
    )
    with pytest.raises(ScientificProvenanceError, match="differs from implementation HEAD"):
        assert_autopsy_replay_producer_reusable(complete, expected_commit="bbb222")


def test_nested_router_binds_expert_git_identity() -> None:
    source = (ROOT / "scripts" / "run_scientific_recovery_v8_nested_router.py").read_text(
        encoding="utf-8"
    )
    assert "assert_router_expert_reusable" in source
    aggregator = (ROOT / "scripts" / "aggregate_scientific_recovery_v8_router.py").read_text(
        encoding="utf-8"
    )
    assert "assert_router_expert_reusable" in aggregator
    assert "train" in source and "summary.json" in source
    assert "_merged_inner_oof_git_identity" in source
    assert '"git_dirty": git_identity["git_dirty"]' in source
    assert "_bind_expert_routing_ttc" in source
    assert "_load_signed_trainer_predictions" in source
    assert "_frozen_router_config_sha256_by_fold" in aggregator
    assert 'frozen.manifest["c1_analysis_plans"]["router_regime"]' not in aggregator
    assert "integral_event_count_from_reconstructed" in source
    assert "_repo_relative" in aggregator
    assert "combined.relative_to(ROOT)" not in aggregator
    expert_source = (ROOT / "scripts" / "train_scientific_recovery_v8_router_expert.py").read_text(
        encoding="utf-8"
    )
    assert "integral_event_count_from_reconstructed" in expert_source


def test_canonical_orchestrators_fail_closed_on_bypass_env_and_unsigned_aggregate() -> None:
    all_trainings = (ROOT / "scripts" / "run_scientific_recovery_v8_all_trainings.ps1").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts" / "run_scientific_recovery_v8.ps1").read_text(encoding="utf-8")
    assert "scientific execution forbids bypass environment variable" in all_trainings
    assert "scientific execution forbids bypass environment variable" in runner
    assert "catch { }" not in runner
    assert "signed aggregate_seed7 reuse state is unreadable" in runner


def test_canonical_runner_contains_no_bypass_environment_variables() -> None:
    runners = [
        ROOT / "scripts" / "run_scientific_recovery_v8_all_trainings.ps1",
        ROOT / "scripts" / "run_scientific_recovery_v8.ps1",
    ]
    for path in runners:
        text = path.read_text(encoding="utf-8")
        for name in ("DINO_NUM_CHUNKS", "DINO_ALLOW_PARTIAL_CACHE", "ALLOW_DIRTY_MATERIALIZE"):
            assert f"$env:{name}" not in text
            assert f"{name}=" not in text or name == "PYTHONUTF8"


def test_scientific_bypass_env_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in FORBIDDEN_SCIENTIFIC_ENV:
        monkeypatch.delenv(name, raising=False)
    refuse_scientific_bypass_env()
    monkeypatch.setenv("DINO_ALLOW_PARTIAL_CACHE", "1")
    with pytest.raises(ScientificProvenanceError, match="DINO_ALLOW_PARTIAL_CACHE"):
        refuse_scientific_bypass_env()


def test_sealed_evaluation_cannot_be_selected_by_canonical_runners() -> None:
    for path in (
        ROOT / "scripts" / "run_scientific_recovery_v8_all_trainings.ps1",
        ROOT / "scripts" / "run_scientific_recovery_v8.ps1",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "codabench" not in text or "closed" in text
        assert "-stage public" not in text
        assert "private test" not in text or "closed" in text
