from __future__ import annotations

import torch

from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    ObjectJEPAConfig,
    geometric_dynamics_targets,
    inverse_ttc_distribution_to_seconds,
    object_event_jepa_loss,
    roi_sample,
)


def _fixture() -> tuple[
    ObjectCentricEventJEPA,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(7)
    config = ObjectJEPAConfig(
        in_channels=4,
        action_dim=3,
        embedding_dim=48,
        feature_dim=32,
        predictor_depth=1,
        predictor_heads=6,
        dropout=0.0,
    )
    model = ObjectCentricEventJEPA(config)
    batch, context_steps, horizons, objects = 2, 3, 3, 2
    context_events = torch.randn(batch, context_steps, 4, 32, 32)
    context_boxes = torch.tensor(
        [
            [
                [[0.10, 0.10, 0.40, 0.50], [0.50, 0.20, 0.80, 0.60]],
                [[0.09, 0.09, 0.41, 0.51], [0.49, 0.19, 0.81, 0.61]],
                [[0.08, 0.08, 0.42, 0.52], [0.48, 0.18, 0.82, 0.62]],
            ]
        ]
        * batch
    )
    context_mask = torch.ones(batch, context_steps, objects, dtype=torch.bool)
    future_events = torch.randn(batch, horizons, 4, 32, 32)
    future_boxes = context_boxes[:, :horizons].clone()
    future_boxes[..., :2] -= 0.01
    future_boxes[..., 2:] += 0.01
    future_mask = torch.ones(batch, horizons, objects, dtype=torch.bool)
    horizon_values = torch.tensor([0.1, 0.25, 0.5])
    return (
        model,
        context_events,
        context_boxes,
        context_mask,
        future_events,
        future_boxes,
        future_mask,
        horizon_values,
    )


def test_roi_sample_uses_normalized_box_coordinates() -> None:
    feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    sampled = roi_sample(feature, boxes, output_size=2)

    assert sampled.shape == (1, 2, 1, 2, 2)
    assert torch.equal(sampled[0, 0, 0], torch.tensor([[0.0, 3.0], [12.0, 15.0]]))
    assert torch.all(sampled[0, 1] == 0.0)


def test_object_jepa_shapes_gradients_and_teacher_isolation() -> None:
    (
        model,
        context_events,
        context_boxes,
        context_mask,
        future_events,
        future_boxes,
        future_mask,
        horizons,
    ) = _fixture()
    action_a = torch.zeros(2, 3, 3)
    action_b = torch.ones(2, 3, 3)
    action_mask = torch.ones(2, 3, dtype=torch.bool)

    output_a = model(
        context_events,
        context_boxes,
        context_mask,
        future_events,
        future_boxes,
        future_mask,
        horizons,
        future_ego_actions=action_a,
        future_ego_action_mask=action_mask,
    )
    output_b = model(
        context_events,
        context_boxes,
        context_mask,
        future_events,
        future_boxes,
        future_mask,
        horizons,
        future_ego_actions=action_b,
        future_ego_action_mask=action_mask,
    )

    assert output_a.predicted_latents.shape == (2, 3, 2, 48)
    assert output_a.target_latents.shape == (2, 3, 2, 48)
    assert torch.equal(output_a.target_latents, output_b.target_latents)
    assert not torch.allclose(output_a.predicted_latents, output_b.predicted_latents)
    assert output_a.action_conditioning_mask.all()
    geometry = geometric_dynamics_targets(context_boxes[:, -1], future_boxes, horizons)
    losses = object_event_jepa_loss(
        output_a,
        geometry,
        ttc_target_s=torch.full((2, 2), 2.0),
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_target_ema_and_inverse_ttc_conversion() -> None:
    model, *_unused = _fixture()
    with torch.no_grad():
        first_context = next(model.context_encoder.parameters())
        first_target = next(model.target_encoder.parameters())
        first_context.add_(1.0)
        before = first_target.clone()
        model.update_target_encoder(0.5)
        assert torch.allclose(first_target, before + 0.5)

    mean, standard_deviation = inverse_ttc_distribution_to_seconds(
        torch.tensor([0.5, -0.25]),
        torch.log(torch.tensor([0.01, 0.04])),
    )
    assert torch.allclose(mean, torch.tensor([2.0, -4.0]))
    assert torch.all(standard_deviation > 0)


def test_geometry_targets_use_depth_when_available() -> None:
    current_boxes = torch.tensor([[[0.2, 0.2, 0.4, 0.4]]])
    future_boxes = torch.tensor([[[[0.2, 0.2, 0.5, 0.5]]]])
    targets = geometric_dynamics_targets(
        current_boxes,
        future_boxes,
        torch.tensor([0.5]),
        context_depth_m=torch.tensor([[10.0]]),
        future_depth_m=torch.tensor([[[8.0]]]),
    )

    assert torch.allclose(targets[..., 4], torch.tensor([[[-0.2]]]))
    assert torch.allclose(targets[..., 5], torch.tensor([[[0.4]]]))


def test_recurrence_and_geometry_ablation_flags_change_the_encoder() -> None:
    torch.manual_seed(11)
    base_config = ObjectJEPAConfig(
        in_channels=4,
        action_dim=3,
        embedding_dim=48,
        feature_dim=32,
        predictor_depth=1,
        predictor_heads=6,
        dropout=0.0,
    )
    base = ObjectCentricEventJEPA(base_config)
    no_recurrence = ObjectCentricEventJEPA(
        ObjectJEPAConfig(**{**base_config.__dict__, "use_recurrence": False})
    )
    no_geometry = ObjectCentricEventJEPA(
        ObjectJEPAConfig(**{**base_config.__dict__, "use_geometry": False})
    )
    no_recurrence.load_state_dict(base.state_dict())
    no_geometry.load_state_dict(base.state_dict())
    _, events, boxes, masks, *_unused = _fixture()

    base_output = base.context_encoder(events, boxes, masks)
    no_recurrence_output = no_recurrence.context_encoder(events, boxes, masks)
    no_geometry_output = no_geometry.context_encoder(events, boxes, masks)

    assert not torch.allclose(base_output.object_tokens, no_recurrence_output.object_tokens)
    assert not torch.allclose(base_output.object_tokens, no_geometry_output.object_tokens)
