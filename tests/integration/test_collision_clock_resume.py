from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockOuterDevBatch,
    CollisionClockOuterTrainBatch,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.models.collision_clock_features import (
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_ttc import CollisionClockConfig, X0HeightBypassDirectPhase
from e_jepa_ttc.training.collision_clock_eap import (
    CollisionClockScientificIdentity,
    CollisionClockTrainingConfig,
    train_collision_clock_updates,
    validate_resume_checkpoint,
)


def _model(
    motion_feature_mode: str = "global_uniform",
) -> X0HeightBypassDirectPhase:
    config = CollisionClockConfig(
        encoder_hidden_dim=8,
        encoder_token_dim=4,
        residual_depth=1,
        clock_hidden_dim=4,
        dropout=0.05,
        motion_feature_mode=motion_feature_mode,
    )
    encoder = HeightBypassEndpointEncoder(
        HeightBypassEncoderConfig(
            in_channels=12,
            hidden_dim=8,
            token_dim=4,
            residual_depth=1,
            dropout=0.05,
        )
    )
    return X0HeightBypassDirectPhase(encoder, config)


def _batches() -> list[CollisionClockOuterTrainBatch]:
    generator = torch.Generator().manual_seed(99)
    return [
        CollisionClockOuterTrainBatch(
            inputs=torch.randn(1, 3, 12, 16, 16, generator=generator),
            delta_t_s=torch.full((1, 2), 0.05),
            target_ttc_seconds=torch.tensor([1.0 + index]),
            sample_tokens=(f"token-{index}",),
        )
        for index in range(2)
    ]


def _identity(
    model: X0HeightBypassDirectPhase, **changes: object
) -> CollisionClockScientificIdentity:
    values: dict[str, object] = {
        "git_commit_observed": "1" * 40,
        "git_dirty_observed": False,
        "arm_id": "X0-DYN-U",
        "scientific_role": "candidate",
        "reference_family": None,
        "seed": 7,
        "outer_fold": 0,
        "motion_feature_mode": "global_uniform",
        "model_class": model.__class__.__name__,
        "model_topology_sha256": module_topology_sha256(model),
        "initialization_sha256": tensor_state_sha256(model),
        "config_path": "configs/x0_dyn_u.yaml",
        "config_sha256": "2" * 64,
        "protocol_path": "configs/protocol.json",
        "protocol_sha256": "3" * 64,
        "reference_path": "configs/reference.json",
        "reference_sha256": "4" * 64,
        "split_manifest_path": "data/split.json",
        "split_manifest_sha256": "5" * 64,
        "cache_manifest_path": "cache/manifest.json",
        "cache_manifest_sha256": "6" * 64,
        "ordered_token_identity_sha256": "7" * 64,
        "target_sha256": "8" * 64,
        "fold_assignment_sha256": "9" * 64,
        "sample_weight_sha256": "a" * 64,
        "train_token_subset_sha256": "b" * 64,
        "dev_token_subset_sha256": "c" * 64,
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "scheduler_config": {"name": "constant"},
        "precision_mode": "float32",
        "update_budget": 4,
        "checkpoint_policy": "last_update_fixed_budget",
    }
    values.update(changes)
    return CollisionClockScientificIdentity(**values)  # type: ignore[arg-type]


def test_continuous_n_equals_k_load_n_minus_k(tmp_path: Path) -> None:
    config = CollisionClockTrainingConfig(arm_id="X0-DYN-U", update_budget=4)
    torch.manual_seed(7)
    continuous = _model()
    identity = _identity(continuous)
    continuous_result = train_collision_clock_updates(
        continuous,
        _batches(),
        config=config,
        scientific_identity=identity,
        checkpoint_path=tmp_path / "continuous.pt",
    )

    torch.manual_seed(7)
    partial = _model()
    partial_identity = _identity(partial)
    train_collision_clock_updates(
        partial,
        _batches(),
        config=config,
        scientific_identity=partial_identity,
        checkpoint_path=tmp_path / "resumed.pt",
        stop_after_updates=2,
    )
    torch.manual_seed(7)
    resumed = _model()
    resumed_identity = _identity(resumed)
    resumed_result = train_collision_clock_updates(
        resumed,
        _batches(),
        config=config,
        scientific_identity=resumed_identity,
        checkpoint_path=tmp_path / "resumed.pt",
        resume=True,
    )
    assert continuous_result.losses == resumed_result.losses
    assert continuous_result.checkpoint_frozen is True
    assert continuous_result.batch_schedule_sha256 == resumed_result.batch_schedule_sha256
    for name, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0.0, atol=0.0)


def _partial_checkpoint(
    tmp_path: Path,
    *,
    arm_id: str = "X0-DYN-U",
    motion_feature_mode: str = "global_uniform",
) -> tuple[Path, CollisionClockScientificIdentity]:
    torch.manual_seed(7)
    model = _model(motion_feature_mode)
    config = CollisionClockTrainingConfig(arm_id=arm_id, update_budget=2)
    identity = _identity(
        model,
        arm_id=arm_id,
        motion_feature_mode=motion_feature_mode,
        update_budget=2,
    )
    path = tmp_path / f"{arm_id}.pt"
    train_collision_clock_updates(
        model,
        _batches(),
        config=config,
        scientific_identity=identity,
        checkpoint_path=path,
        stop_after_updates=1,
    )
    return path, identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outer_fold", 1),
        ("config_sha256", "d" * 64),
        ("protocol_sha256", "d" * 64),
        ("cache_manifest_sha256", "d" * 64),
        ("split_manifest_sha256", "d" * 64),
        ("git_commit_observed", "d" * 40),
        ("model_topology_sha256", "d" * 64),
        ("train_token_subset_sha256", "d" * 64),
        ("dev_token_subset_sha256", "d" * 64),
        ("target_sha256", "d" * 64),
        ("fold_assignment_sha256", "d" * 64),
    ],
)
def test_resume_rejects_every_scientific_identity_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    path, identity = _partial_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        validate_resume_checkpoint(path, expected_identity=replace(identity, **{field: value}))


def test_resume_rejects_dyn_checkpoint_as_base_and_base_as_dyn(tmp_path: Path) -> None:
    dyn_path, dyn_identity = _partial_checkpoint(tmp_path, arm_id="X0-DYN-U")
    base_path, base_identity = _partial_checkpoint(
        tmp_path,
        arm_id="X0-BASE-U",
        motion_feature_mode="global_uniform_zeroed_control",
    )
    expected_base = replace(
        dyn_identity,
        arm_id="X0-BASE-U",
        motion_feature_mode="global_uniform_zeroed_control",
    )
    expected_dyn = replace(
        base_identity,
        arm_id="X0-DYN-U",
        motion_feature_mode="global_uniform",
    )
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        validate_resume_checkpoint(dyn_path, expected_identity=expected_base)
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        validate_resume_checkpoint(base_path, expected_identity=expected_dyn)


def test_resume_rejects_seed_and_dirty_worktree_identity() -> None:
    torch.manual_seed(7)
    model = _model()
    with pytest.raises(ValueError, match="arm/seed/fold"):
        _identity(model, seed=23)
    with pytest.raises(ValueError, match="dirty worktree"):
        _identity(model, git_dirty_observed=True)


def test_resume_rejects_truncated_checkpoint(tmp_path: Path) -> None:
    path, identity = _partial_checkpoint(tmp_path)
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError, match="truncated"):
        validate_resume_checkpoint(path, expected_identity=identity)


def test_resume_rejects_incorrect_checkpoint_sha(tmp_path: Path) -> None:
    path, identity = _partial_checkpoint(tmp_path)
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_file_sha256"] = "0" * 64
    sign_artifact(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="physical SHA"):
        validate_resume_checkpoint(path, expected_identity=identity)


def test_trainer_rejects_outer_dev_batch(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = _model()
    config = CollisionClockTrainingConfig(arm_id="X0-DYN-U", update_budget=1)
    identity = _identity(model, update_budget=1)
    dev = CollisionClockOuterDevBatch(
        inputs=torch.randn(1, 3, 12, 16, 16),
        delta_t_s=torch.full((1, 2), 0.05),
        target_ttc_seconds=torch.tensor([1.0]),
        sample_tokens=("token",),
        sequence_ids=("sequence",),
        track_ids=("track",),
        outer_fold=0,
        sample_weights=torch.tensor([1.0], dtype=torch.float64),
    )
    with pytest.raises(TypeError, match="OuterTrainBatch"):
        train_collision_clock_updates(
            model,
            [dev],  # type: ignore[list-item]
            config=config,
            scientific_identity=identity,
            checkpoint_path=tmp_path / "forbidden.pt",
        )
