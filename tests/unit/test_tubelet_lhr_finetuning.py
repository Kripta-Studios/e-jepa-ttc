from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig
from e_jepa_ttc.training.tubelet_finetuning import (
    TubeletOptimizationConfig,
    apply_optimizer_phase,
    build_tubelet_optimizer,
    checkpoint_is_eligible,
    is_prediction_collapsed,
    prediction_health,
    resolve_optimization_config,
    split_parameter_groups,
    validate_optimizer_manifest,
)


def _model(*, pooling: str = "query") -> EJEPATubeletLHR:
    return EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=16,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=False,
            pooling=pooling,
            query_count=2,
        )
    )


def _optimization_config(*, threshold: float = 0.01) -> TubeletOptimizationConfig:
    return TubeletOptimizationConfig(
        backbone_learning_rate=1e-5,
        pooling_learning_rate=1e-4,
        head_learning_rate=3e-4,
        warmup_pooling_learning_rate=3e-4,
        warmup_head_learning_rate=1e-3,
        backbone_weight_decay=0.01,
        readout_weight_decay=0.0,
        readout_warmup_optimizer_steps=2,
        min_prediction_std_ratio=threshold,
        collapse_patience=3,
    )


def test_optimizer_groups_are_disjoint_and_exhaustive() -> None:
    model = _model()
    groups = split_parameter_groups(model)
    grouped = [(name, parameter) for values in groups.values() for name, parameter in values]
    grouped_names = {name for name, _ in grouped}
    grouped_ids = {id(parameter) for _, parameter in grouped}
    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("collision_head.")
    }
    assert grouped_names == expected
    assert len(grouped_ids) == len(grouped)


def test_collision_head_is_not_optimized_without_collision_loss() -> None:
    model = _model()
    groups = split_parameter_groups(model)
    optimized = {id(parameter) for values in groups.values() for _, parameter in values}
    assert all(id(parameter) not in optimized for parameter in model.collision_head.parameters())


def test_legacy_config_falls_back_to_single_learning_rate() -> None:
    legacy = SimpleNamespace(learning_rate=3e-4, weight_decay=0.01)
    resolved = resolve_optimization_config(legacy)
    assert resolved.backbone_learning_rate == pytest.approx(3e-4)
    assert resolved.pooling_learning_rate == pytest.approx(3e-4)
    assert resolved.head_learning_rate == pytest.approx(3e-4)
    assert resolved.backbone_weight_decay == pytest.approx(0.01)
    assert resolved.readout_weight_decay == pytest.approx(0.01)
    assert resolved.readout_warmup_optimizer_steps == 0


def test_warmup_sets_backbone_lr_to_zero() -> None:
    model = _model()
    config = _optimization_config()
    optimizer, _ = build_tubelet_optimizer(model, config)
    phase, rates = apply_optimizer_phase(optimizer, 0, config)
    assert phase == "readout_warmup"
    assert rates == pytest.approx({"backbone": 0.0, "pooling": 3e-4, "ttc_head": 1e-3})


def test_post_warmup_restores_discriminative_learning_rates() -> None:
    model = _model()
    config = _optimization_config()
    optimizer, _ = build_tubelet_optimizer(model, config)
    phase, rates = apply_optimizer_phase(optimizer, 2, config)
    assert phase == "full_finetune"
    assert rates == pytest.approx({"backbone": 1e-5, "pooling": 1e-4, "ttc_head": 3e-4})


def test_mean_pooling_allows_empty_pooling_group() -> None:
    model = _model(pooling="mean")
    groups = split_parameter_groups(model)
    assert groups["pooling"] == []
    optimizer, manifest = build_tubelet_optimizer(model, _optimization_config())
    assert [group["name"] for group in optimizer.param_groups] == ["backbone", "ttc_head"]
    assert [entry["name"] for entry in manifest["groups"]] == ["backbone", "ttc_head"]


def test_constant_predictions_are_marked_collapsed() -> None:
    config = _optimization_config(threshold=0.05)
    health = prediction_health([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert health["prediction_std_ratio"] == 0.0
    assert is_prediction_collapsed(health, config)
    assert not checkpoint_is_eligible(
        score=1.0,
        health=health,
        optimizer_step=3,
        config=config,
    )


def test_variable_predictions_pass_collapse_gate() -> None:
    config = _optimization_config(threshold=0.05)
    health = prediction_health([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert not is_prediction_collapsed(health, config)
    assert checkpoint_is_eligible(
        score=1.0,
        health=health,
        optimizer_step=3,
        config=config,
    )


def test_prediction_health_matches_known_values() -> None:
    health = prediction_health(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 2.0, 4.0]),
    )
    assert health["sample_count"] == 3
    assert health["finite_sample_count"] == 3
    assert health["mae"] == pytest.approx(1.0 / 3.0)
    assert health["prediction_min"] == pytest.approx(1.0)
    assert health["prediction_max"] == pytest.approx(4.0)
    assert health["pearson"] is not None
    assert health["pearson"] > 0.98


def test_optimizer_group_manifest_is_resume_stable() -> None:
    config = _optimization_config()
    _, first = build_tubelet_optimizer(_model(), config)
    _, second = build_tubelet_optimizer(_model(), config)
    validate_optimizer_manifest(first, second)
    altered = dict(second)
    altered["artifact_type"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        validate_optimizer_manifest(first, altered)
