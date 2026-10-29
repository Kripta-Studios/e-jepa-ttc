from __future__ import annotations

import torch

from e_jepa_ttc.losses.level_dynamics_jepa import build_temporal_residual_target, dense_cosine_loss
from e_jepa_ttc.models.dense_level_dynamics_jepa import (
    DenseLevelDynamicsConfig,
    DenseLevelDynamicsJEPA,
)
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHRConfig


def _config() -> DenseLevelDynamicsConfig:
    return DenseLevelDynamicsConfig(
        encoder=EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=16,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
        ),
        projection_dim=16,
        predictor_dim=16,
        predictor_layers=1,
        predictor_heads=4,
        predictor_mlp_ratio=2,
        patch_query_chunk_size=4,
        max_temporal_steps=3,
        max_patches=16,
        max_horizons=3,
        ema_total_updates=4,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4)
    return (
        torch.randn(1, 3, 2, 16, 16),
        torch.randn(1, 3, 3, 2, 16, 16),
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    )


def test_target_is_exact_frozen_eval_only_and_ema_aligned() -> None:
    model = DenseLevelDynamicsJEPA(_config())
    online_state = model.online_representation.state_dict()
    target_state = model.target_representation.state_dict()
    assert online_state.keys() == target_state.keys()
    assert all(torch.equal(online_state[key], target_state[key]) for key in online_state)
    assert not any(
        parameter.requires_grad for parameter in model.target_representation.parameters()
    )

    model.train()
    assert model.online_representation.training
    assert not model.target_representation.training
    name, parameter = next(iter(model.online_representation.named_parameters()))
    before = dict(model.target_representation.named_parameters())[name].detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)
    momentum = model.update_target_ema(0, total_updates=4)
    after = dict(model.target_representation.named_parameters())[name]
    assert momentum == 0.99
    assert torch.allclose(after, before * momentum + parameter * (1.0 - momentum))
    assert model.ema_momentum(3, total_updates=4) == 0.9999


def test_dense_shapes_delta_t_conditioning_and_expected_head_gradients() -> None:
    model = DenseLevelDynamicsJEPA(_config()).eval()
    context, future, delta = _inputs()
    output = model(context, future, delta)
    changed = model(context, future, torch.tensor([[0.15, 0.25, 0.35]]))
    changed_target_content = model(context, future + 50.0, delta)

    assert output.level_tokens.shape == (1, 3, 4, 16)
    assert output.dynamics_tokens.shape == (1, 3, 4, 16)
    assert output.predicted_level_tokens.shape == (1, 3, 4, 16)
    assert output.predicted_dynamics_tokens.shape == (1, 3, 4, 16)
    assert output.valid_target_patch_mask.shape == (1, 3, 4)
    assert output.patch_coordinates.shape == (4, 2)
    assert not torch.allclose(output.predicted_level_tokens, changed.predicted_level_tokens)
    assert torch.allclose(
        output.predicted_level_tokens,
        changed_target_content.predicted_level_tokens,
    )
    assert torch.allclose(
        output.predicted_dynamics_tokens,
        changed_target_content.predicted_dynamics_tokens,
    )
    assert not output.target_level_tokens.requires_grad
    assert not output.target_dynamics_tokens.requires_grad

    model.train()
    output = model(context, future, delta)
    loss = dense_cosine_loss(
        output.predicted_level_tokens,
        output.target_level_tokens,
        output.valid_target_patch_mask,
    )
    loss.backward()
    assert model.online_representation.level_head.projection.weight.grad is not None
    assert model.online_representation.dynamics_head.projection.weight.grad is None
    assert not any(
        parameter.grad is not None for parameter in model.target_representation.parameters()
    )

    model.zero_grad(set_to_none=True)
    output = model(context, future, delta)
    residual = build_temporal_residual_target(
        output.target_reference_dynamics_tokens,
        output.target_dynamics_tokens,
        output.target_reference_valid_patch_mask,
        output.valid_target_patch_mask,
    )
    residual_loss = dense_cosine_loss(
        output.predicted_residual_tokens,
        residual.tokens,
        residual.valid_mask,
    )
    residual_loss.backward()
    assert model.online_representation.dynamics_head.projection.weight.grad is not None
    assert model.predictor.residual_projection[1].weight.grad is not None


def test_resource_contract_rejects_unprofiled_shape() -> None:
    config = _config()
    model = DenseLevelDynamicsJEPA(config)
    context = torch.randn(3, 3, 2, 16, 16)
    future = torch.randn(3, 3, 3, 2, 16, 16)
    delta = torch.full((3, 3), 0.1)

    try:
        model(context, future, delta)
    except ValueError as exc:
        assert "runtime_batch" in str(exc)
    else:  # pragma: no cover - documents the non-permissive residency boundary
        raise AssertionError("resident batch limit must fail before encoder allocation")
