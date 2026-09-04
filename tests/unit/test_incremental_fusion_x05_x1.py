from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.artifacts.hashing import compute_artifact_hash, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    canonical_records_hash,
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.evaluation.incremental_fusion import (
    DYNAMIC_SLOT_NAMES,
    FEATURE_COLUMNS,
    _ridge_fit,
    _ridge_predict,
    deterministic_within_sequence_shuffle,
    evaluate_x1_gate,
    run_x05_cross_fit,
    validate_feature_table,
)
from e_jepa_ttc.models.collision_clock_math import (
    benchmark_phase_to_inverse_ttc,
    phase_lower_bound,
)
from e_jepa_ttc.models.incremental_residual import (
    FrozenA5DynamicResidualAdapter,
    add_safe_phase_residual,
)
from e_jepa_ttc.training.incremental_residual import (
    X1TrainingConfig,
    X1TrainingIdentity,
    deterministic_sequence_grouped_schedule,
    normalization_sha256,
    train_x1_fixed_budget,
    trainable_mask_sha256,
)
from scripts.run_scientific_recovery_v9_eclock_x05_x1 import _benchmark_x1_devices


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _small_feature_frame() -> pd.DataFrame:
    rows = []
    targets = (1.5, 4.5, 8.0, -4.0)
    for sequence_index in range(9):
        sequence = f"sequence-{sequence_index}"
        fold = sequence_index // 3
        for repeat in range(2):
            for bucket_index, target in enumerate(targets):
                phase = -np.log1p(-0.1 / target)
                dynamic = np.asarray(
                    [phase * (index + 1) + sequence_index * 0.001 for index in range(9)],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "sample_token": f"{sequence}-{repeat}-{bucket_index}",
                        "sequence_id": sequence,
                        "track_id": f"track-{repeat}",
                        "outer_fold": fold,
                        "target_ttc_s": target,
                        "target_benchmark_phase": phase,
                        "sample_weight": 1.0,
                        "a5_predicted_benchmark_phase": phase + 0.01,
                        "base_predicted_benchmark_phase": phase + 0.02,
                        "dyn_predicted_benchmark_phase": phase + 0.015,
                        "pair_predicted_benchmark_phase": phase + 0.009,
                        **dict(zip(DYNAMIC_SLOT_NAMES, dynamic, strict=True)),
                    }
                )
    return pd.DataFrame(rows)


def _production_feature_frame() -> tuple[pd.DataFrame, dict, dict]:
    base = _small_feature_frame()
    repeats = []
    for repeat in range(114):
        value = base.copy()
        value["sample_token"] = value["sample_token"].astype(str) + f"-r{repeat}"
        repeats.append(value)
    frame = pd.concat(repeats, ignore_index=True).iloc[:8192].copy()
    # Force exact canonical sequence-to-fold coverage after the truncation.
    frame["transport_valid"] = True
    for column in (
        "a5_checkpoint_sha256",
        "base_checkpoint_sha256",
        "dyn_checkpoint_sha256",
        "pair_checkpoint_sha256",
    ):
        frame[column] = "a" * 64
    frame["x0_protocol_sha256"] = "b" * 64
    frame["x0_reference_sha256"] = "c" * 64
    frame["cache_manifest_sha256"] = "d" * 64
    frame["split_manifest_sha256"] = "e" * 64
    frame = frame.loc[:, FEATURE_COLUMNS]
    protocol = {
        "artifact_sha256": "b" * 64,
        "canonical_sequence_ids": sorted(frame["sequence_id"].unique()),
        "canonical_sequence_to_fold": {
            sequence: int(frame.loc[frame["sequence_id"] == sequence, "outer_fold"].iloc[0])
            for sequence in frame["sequence_id"].unique()
        },
        "cache_binding": {"file_sha256": "d" * 64},
        "split_binding": {"file_sha256": "e" * 64},
    }
    protocol["canonical_hashes"] = {
        "token_identity_sha256": canonical_records_hash(
            frame, ("sample_token", "sequence_id", "track_id")
        ),
        "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc_s")),
        "fold_assignment_sha256": canonical_records_hash(
            frame, ("sample_token", "sequence_id", "outer_fold")
        ),
        "sample_weight_sha256": canonical_records_hash(frame, ("sample_token", "sample_weight")),
    }
    replay = {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "identity_hashes_exact": True,
        "replay_matches_x0": True,
        "target_not_passed_to_extractor": True,
        "sealed_evaluation_opened": False,
    }
    return frame, protocol, replay


def test_protocol_is_signed_and_draft_202012(repo_root: Path) -> None:
    path = repo_root / "configs/protocol/scientific_recovery_v9_eclock_x05_x1.json"
    schema_path = repo_root / "schemas/scientific_recovery_v9_eclock_x05_x1_protocol_v1.schema.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert verify_artifact_hash(payload)
    assert schema["$schema"].endswith("draft/2020-12/schema")
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_protocol_hash_rejects_mutation(repo_root: Path) -> None:
    payload = json.loads(
        (repo_root / "configs/protocol/scientific_recovery_v9_eclock_x05_x1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["production_row_count"] = 8191
    assert not verify_artifact_hash(payload)
    assert compute_artifact_hash(payload) != payload["artifact_sha256"]


def test_exact_nine_slot_order_is_frozen() -> None:
    assert DYNAMIC_SLOT_NAMES == (
        "translation_x",
        "translation_y",
        "divergence_x",
        "divergence_y",
        "divergence_isotropic",
        "flow_magnitude",
        "confidence_margin",
        "entropy",
        "cycle_error",
    )


def test_x1_device_benchmark_uses_only_valid_synthetic_phases() -> None:
    result = _benchmark_x1_devices()
    assert result["selected_device"] in {"cpu", "cuda:0"}
    assert result["scientific_rows_observed"] is False
    assert result["synthetic_rows"] == 8192


def test_zero_initialization_exactly_replays_a5() -> None:
    model = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    phase = torch.tensor([-0.2, -0.01, 0.0, 0.2], dtype=torch.float32)
    slots = torch.randn(4, 9)
    assert torch.equal(model(phase, slots), phase)


def test_safe_phase_residual_preserves_domain_and_sign_coordinates() -> None:
    lower = phase_lower_bound(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1)
    a5 = torch.tensor([lower + 1e-4, -0.2, 0.0, 0.4], dtype=torch.float64)
    raw = torch.tensor([-100.0, -2.0, 0.0, 3.0], dtype=torch.float64)
    result = add_safe_phase_residual(
        a5,
        raw,
        metric_delta_t_s=0.1,
        minimum_abs_prediction_ttc_s=0.1,
    )
    assert torch.isfinite(result).all()
    assert (result >= lower).all()
    inverse = benchmark_phase_to_inverse_ttc(result, metric_delta_t_s=0.1)
    assert torch.isfinite(inverse).all()
    assert torch.equal(result[2:3], a5[2:3])


def test_safe_phase_residual_rejects_invalid_a5() -> None:
    lower = phase_lower_bound(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1)
    with pytest.raises(ValueError, match="outside"):
        add_safe_phase_residual(
            torch.tensor([lower - 0.01]),
            torch.zeros(1),
            metric_delta_t_s=0.1,
            minimum_abs_prediction_ttc_s=0.1,
        )


def test_shuffle_is_within_sequence_reproducible_and_row_complete() -> None:
    slots = np.arange(108, dtype=np.float64).reshape(12, 9)
    sequences = ["a"] * 6 + ["b"] * 6
    first, first_sha = deterministic_within_sequence_shuffle(
        slots, sequences, seed=20260904, outer_fold=0, partition="meta-train"
    )
    second, second_sha = deterministic_within_sequence_shuffle(
        slots, sequences, seed=20260904, outer_fold=0, partition="meta-train"
    )
    assert np.array_equal(first, second)
    assert first_sha == second_sha
    assert {tuple(row) for row in first[:6]} == {tuple(row) for row in slots[:6]}
    assert {tuple(row) for row in first[6:]} == {tuple(row) for row in slots[6:]}


def test_shuffle_partitions_have_distinct_signed_permutations() -> None:
    slots = np.arange(180, dtype=np.float64).reshape(20, 9)
    sequences = ["a"] * 10 + ["b"] * 10
    _, train_sha = deterministic_within_sequence_shuffle(
        slots, sequences, seed=20260904, outer_fold=1, partition="meta-train"
    )
    _, test_sha = deterministic_within_sequence_shuffle(
        slots, sequences, seed=20260904, outer_fold=1, partition="meta-test"
    )
    assert train_sha != test_sha


def test_weighted_ridge_is_float64_deterministic() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(128, 10)).astype(np.float64)
    y = x[:, 0] - 0.2 * x[:, 1]
    weights = np.linspace(0.5, 1.5, 128, dtype=np.float64)
    first = _ridge_fit(x, y, weights, 1e-4)
    second = _ridge_fit(x, y, weights, 1e-4)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert np.isfinite(_ridge_predict(x, *first)).all()


def test_x05_crossfit_never_uses_meta_test_for_fit_or_selection(tmp_path: Path) -> None:
    protocol = {
        "bootstrap": {
            "seed": 20260814,
            "draws": 50,
            "method": "paired_hierarchical_sequence_then_track_cluster_bootstrap",
        },
        "artifact_sha256": "f" * 64,
    }
    gate = run_x05_cross_fit(
        _small_feature_frame(), x0_protocol=protocol, output_root=tmp_path / "x05"
    )
    assert verify_artifact_hash(gate)
    summary = json.loads((tmp_path / "x05/x05_meta_fold_summary.json").read_text())
    for fold in summary["folds"]:
        for arm in fold["arms"].values():
            assert arm["outer_meta_test_used_for_fit_or_selection"] is False
            assert not set(arm["meta_train_sequences"]) & set(arm["meta_test_sequences"])


@pytest.mark.parametrize("mutation", ["duplicate", "target", "fold", "reorder", "nonfinite"])
def test_feature_table_fault_injection_fails_closed(mutation: str) -> None:
    frame, protocol, replay = _production_feature_frame()
    corrupted = frame.copy()
    if mutation == "duplicate":
        corrupted.loc[1, "sample_token"] = corrupted.loc[0, "sample_token"]
    elif mutation == "target":
        corrupted.loc[0, "target_ttc_s"] += 0.1
    elif mutation == "fold":
        corrupted.loc[0, "outer_fold"] = (int(corrupted.loc[0, "outer_fold"]) + 1) % 3
    elif mutation == "reorder":
        corrupted = corrupted.loc[:, list(reversed(corrupted.columns))]
    else:
        corrupted.loc[0, DYNAMIC_SLOT_NAMES[0]] = np.nan
    with pytest.raises(ValueError):
        validate_feature_table(corrupted, x0_protocol=protocol, replay_manifest=replay)


def test_feature_gate_rejects_incomplete_replay() -> None:
    frame, protocol, replay = _production_feature_frame()
    replay["replay_matches_x0"] = False
    with pytest.raises(ValueError, match="replay_matches_x0"):
        validate_feature_table(frame, x0_protocol=protocol, replay_manifest=replay)


def test_sequence_grouped_schedule_is_matched_and_deterministic() -> None:
    sequences = [f"s{i // 32}" for i in range(256)]
    tokens = [f"t{i}" for i in range(256)]
    first, first_sha = deterministic_sequence_grouped_schedule(sequences, tokens, seed=7)
    second, second_sha = deterministic_sequence_grouped_schedule(sequences, tokens, seed=7)
    assert first_sha == second_sha
    assert len(first) == 1000
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))


def _training_identity(
    model: FrozenA5DynamicResidualAdapter, schedule_sha: str
) -> X1TrainingIdentity:
    return X1TrainingIdentity(
        training_commit="1" * 40,
        feature_table_sha256="2" * 64,
        x05_gate_sha256="3" * 64,
        protocol_sha256="4" * 64,
        config_sha256="5" * 64,
        arm_id="X1-A5-DYN-U",
        seed=7,
        outer_fold=0,
        train_token_sha256="6" * 64,
        dev_token_sha256="7" * 64,
        topology_sha256=module_topology_sha256(model),
        initialization_sha256=tensor_state_sha256(model),
        trainable_mask_sha256=trainable_mask_sha256(model),
        normalization_sha256=normalization_sha256(np.zeros(9), np.ones(9)),
        batch_schedule_sha256=schedule_sha,
        a5_frozen=True,
        transport_extractor_frozen=True,
        outer_dev_available_to_trainer=False,
    )


def test_x1_resume_matches_continuous_tensor_state_and_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    phase = rng.normal(scale=0.02, size=256).astype(np.float32)
    slots = rng.normal(size=(256, 9)).astype(np.float32)
    target = (phase + 0.01 * slots[:, 0]).astype(np.float32)
    weights = np.ones(256, dtype=np.float32)
    sequences = [f"s{i // 32}" for i in range(256)]
    tokens = [f"t{i}" for i in range(256)]
    schedule, schedule_sha = deterministic_sequence_grouped_schedule(sequences, tokens, seed=7)

    torch.manual_seed(7)
    continuous = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    identity = _training_identity(continuous, schedule_sha)
    train_x1_fixed_budget(
        continuous,
        a5_phase=phase,
        slots=slots,
        target_phase=target,
        sample_weights=weights,
        schedule=schedule,
        config=X1TrainingConfig(arm_id="X1-A5-DYN-U", seed=7, outer_fold=0),
        identity=identity,
        output_root=tmp_path / "continuous",
        device=torch.device("cpu"),
        stop_after_updates=20,
    )

    torch.manual_seed(7)
    resumed = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    train_x1_fixed_budget(
        resumed,
        a5_phase=phase,
        slots=slots,
        target_phase=target,
        sample_weights=weights,
        schedule=schedule,
        config=X1TrainingConfig(arm_id="X1-A5-DYN-U", seed=7, outer_fold=0),
        identity=identity,
        output_root=tmp_path / "resumed",
        device=torch.device("cpu"),
        stop_after_updates=10,
    )
    torch.manual_seed(7)
    resumed_again = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    train_x1_fixed_budget(
        resumed_again,
        a5_phase=phase,
        slots=slots,
        target_phase=target,
        sample_weights=weights,
        schedule=schedule,
        config=X1TrainingConfig(arm_id="X1-A5-DYN-U", seed=7, outer_fold=0),
        identity=identity,
        output_root=tmp_path / "resumed",
        device=torch.device("cpu"),
        resume=True,
        stop_after_updates=20,
    )
    assert tensor_state_sha256(continuous) == tensor_state_sha256(resumed_again)
    continuous_losses = [
        json.loads(line)["loss"]
        for line in (tmp_path / "continuous/progress.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "update"
    ]
    resumed_losses = [
        json.loads(line)["loss"]
        for line in (tmp_path / "resumed/progress.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "update"
    ]
    assert continuous_losses == resumed_losses


def test_truncated_x1_checkpoint_fails_resume(tmp_path: Path) -> None:
    # The resume path must fail before loading when the physical byte hash changes.
    sequences = ["s"] * 256
    tokens = [f"t{i}" for i in range(256)]
    schedule, schedule_sha = deterministic_sequence_grouped_schedule(sequences, tokens, seed=7)
    torch.manual_seed(7)
    model = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    identity = _training_identity(model, schedule_sha)
    arrays = np.zeros(256, dtype=np.float32)
    train_x1_fixed_budget(
        model,
        a5_phase=arrays,
        slots=np.zeros((256, 9), dtype=np.float32),
        target_phase=arrays,
        sample_weights=np.ones(256, dtype=np.float32),
        schedule=schedule,
        config=X1TrainingConfig(arm_id="X1-A5-DYN-U", seed=7, outer_fold=0),
        identity=identity,
        output_root=tmp_path / "run",
        device=torch.device("cpu"),
        stop_after_updates=1,
    )
    checkpoint = tmp_path / "run/resume_latest.pt"
    checkpoint.write_bytes(checkpoint.read_bytes()[:128])
    torch.manual_seed(7)
    fresh = FrozenA5DynamicResidualAdapter(torch.zeros(9), torch.ones(9))
    with pytest.raises(ValueError, match="identity/hash"):
        train_x1_fixed_budget(
            fresh,
            a5_phase=arrays,
            slots=np.zeros((256, 9), dtype=np.float32),
            target_phase=arrays,
            sample_weights=np.ones(256, dtype=np.float32),
            schedule=schedule,
            config=X1TrainingConfig(arm_id="X1-A5-DYN-U", seed=7, outer_fold=0),
            identity=identity,
            output_root=tmp_path / "run",
            device=torch.device("cpu"),
            resume=True,
            stop_after_updates=2,
        )


def _comparison(delta: float, high: float, probability: float) -> dict:
    return {
        "delta_mid": delta,
        "bootstrap": {
            "delta_candidate_minus_reference": {
                "ci95_high": high,
                "probability_delta_lt_zero": probability,
            }
        },
    }


def _integrity() -> dict:
    return {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "coverage_drop_pp": 0.0,
        "a5_replay_exact": True,
        "zero_initialization_replay_exact": True,
        "matched_topology_init_order_budget": True,
        "a5_and_transport_frozen": True,
        "outer_dev_evaluations_per_arm_fold": 1,
        "sealed_evaluation_opened": False,
    }


def test_x1_gate_requires_all_three_positive_gates() -> None:
    comparisons = {
        "dyn_vs_zero": _comparison(-1.1, -0.1, 0.95),
        "dyn_vs_shuffle": _comparison(-0.1, -0.01, 0.95),
        "dyn_vs_a5": _comparison(-3.1, -0.1, 0.95),
    }
    gate = evaluate_x1_gate(comparisons=comparisons, integrity=_integrity())
    assert gate["decision"] == "X1_SEED7_SUPPORTED_REPLICATION_REQUIRED"
    assert gate["replication_authorized"] is True


def test_x1_gate_rejects_incomplete_evidence() -> None:
    integrity = _integrity()
    integrity["outer_dev_evaluations_per_arm_fold"] = 2
    gate = evaluate_x1_gate(
        comparisons={
            "dyn_vs_zero": _comparison(-10, -1, 1),
            "dyn_vs_shuffle": _comparison(-10, -1, 1),
            "dyn_vs_a5": _comparison(-10, -1, 1),
        },
        integrity=integrity,
    )
    assert gate["decision"] == "INVALID_X1"


def test_forbidden_families_remain_closed(repo_root: Path) -> None:
    protocol = json.loads(
        (repo_root / "configs/protocol/scientific_recovery_v9_eclock_x05_x1.json").read_text()
    )
    forbidden = set(protocol["forbidden_execution"])
    assert {"X0-DYN-W", "X2", "X3", "Track-M", "public-validation", "CodaBench"} <= forbidden


def test_powershell_orchestrator_ast_parses(repo_root: Path) -> None:
    script = repo_root / "scripts/run_scientific_recovery_v9_eclock_x05_x1.ps1"
    command = (
        "$e=$null;$t=$null;"
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$t,[ref]$e);"
        "if($e.Count){$e|ForEach-Object{$_.Message};exit 1}"
    )
    subprocess.run(["pwsh", "-NoProfile", "-Command", command], check=True)


def test_no_outer_dev_argument_exists_on_trainer() -> None:
    import inspect

    signature = inspect.signature(train_x1_fixed_budget)
    assert "outer_dev" not in signature.parameters
    assert "validation" not in signature.parameters


def test_target_does_not_enter_transport_extractor_api() -> None:
    import inspect

    from e_jepa_ttc.models.collision_clock_motion import height_free_global_transport_features

    parameters = set(inspect.signature(height_free_global_transport_features).parameters)
    assert parameters == {
        "previous_dense",
        "current_dense",
        "radius",
        "temperature",
        "return_dense_diagnostics",
    }
