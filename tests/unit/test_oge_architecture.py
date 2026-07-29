from __future__ import annotations

from pathlib import Path

import torch

from e_jepa_ttc.geometry import (
    affine_expansion_inverse_ttc,
    area_rate_inverse_ttc,
    height_ratio_inverse_ttc,
)
from e_jepa_ttc.geometry.ego_motion_compensation import CameraEgoMotionCompensator
from e_jepa_ttc.models.attention_residual_router import TaskSpecificAttentionResiduals
from e_jepa_ttc.models.block_causal_transformer import (
    BlockCausalTransformer,
    block_causal_attention_mask,
)
from e_jepa_ttc.models.dense_patch_ttc import BaseEventTubeletBackbone
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCReplica
from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEConfig
from e_jepa_ttc.models.target_query import TargetBackgroundQuery
from e_jepa_ttc.models.temporal_kda import (
    TemporalDeltaMemory,
    kimi_delta_recurrence,
)
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerEncoder
from e_jepa_ttc.teachers.reliability_gated_multiteacher import (
    reliability_gated_teacher_target,
)
from e_jepa_ttc.training.health_monitor import embedding_health
from e_jepa_ttc.training.student_conditioned_curriculum import (
    predicted_roi_probability,
    select_curriculum_boxes,
)


def test_geometry_closed_forms_recover_exponential_scale_rate() -> None:
    times = torch.tensor([0.0, 0.1, 0.2, 0.3])
    rate = 0.4
    height = torch.exp(rate * times)[None, :, None]
    width = torch.exp(rate * times)[None, :, None]
    valid = torch.ones_like(height, dtype=torch.bool)
    height_value, _ = height_ratio_inverse_ttc(height, times, valid_mask=valid)
    area_value, _ = area_rate_inverse_ttc(height * width, times, valid_mask=valid)
    center_x = torch.full_like(width, 0.5)
    center_y = torch.full_like(height, 0.5)
    boxes = torch.stack(
        (
            center_x - width * 0.1,
            center_y - height * 0.1,
            center_x + width * 0.1,
            center_y + height * 0.1,
        ),
        dim=-1,
    )
    affine_value, _ = affine_expansion_inverse_ttc(boxes, times, valid_mask=valid)
    exact_pair_rate = (torch.exp(torch.tensor(rate * 0.1)) - 1.0) / 0.1
    expected = torch.full_like(height_value, exact_pair_rate)
    assert torch.allclose(height_value, expected, atol=1e-5)
    assert torch.allclose(area_value, expected, atol=1e-5)
    assert torch.allclose(affine_value, expected, atol=5e-4)


def test_geometry_closed_forms_estimate_ttc_at_latest_endpoint() -> None:
    times = torch.tensor([0.0, 0.1, 0.2])
    ttc_at_start = 2.0
    remaining = ttc_at_start - times
    height = (1.0 / remaining)[None, :, None]
    valid = torch.ones_like(height, dtype=torch.bool)
    height_value, _ = height_ratio_inverse_ttc(height, times, valid_mask=valid)
    area_value, _ = area_rate_inverse_ttc(height.square(), times, valid_mask=valid)
    boxes = torch.stack(
        (
            0.5 - 0.1 * height,
            0.5 - 0.1 * height,
            0.5 + 0.1 * height,
            0.5 + 0.1 * height,
        ),
        dim=-1,
    )
    affine_value, _ = affine_expansion_inverse_ttc(boxes, times, valid_mask=valid)
    expected = torch.full_like(height_value, 1.0 / remaining[-1])

    assert torch.allclose(height_value, expected, atol=1e-5)
    assert torch.allclose(area_value, expected, atol=1e-5)
    assert torch.allclose(affine_value, expected, atol=5e-4)


def test_camera_translation_compensation_aligns_static_object_and_recovers_ego_ttc() -> None:
    times = torch.tensor([[0.0, 0.1, 0.2]])
    depths = torch.tensor([[[10.2], [10.1], [10.0]]])
    half_extent_m = 0.5
    half_extent = half_extent_m / depths
    boxes = torch.cat(
        (
            0.5 - half_extent,
            0.5 - half_extent,
            0.5 + half_extent,
            0.5 + half_extent,
        ),
        dim=-1,
    ).unsqueeze(2)
    actions = torch.zeros(1, 3, 8)
    actions[..., 3] = 1.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    compensator = CameraEgoMotionCompensator(action_dim=8)

    aligned, yaw, translation, ego_inverse_ttc = compensator(
        boxes,
        depths,
        actions,
        valid,
        times,
        intrinsics_normalized=torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
    )

    expected_box = boxes[:, -1:].expand_as(boxes)
    assert torch.allclose(aligned, expected_box, atol=1e-5)
    assert torch.allclose(yaw, torch.zeros_like(yaw))
    assert torch.allclose(translation[0, :, 2], torch.tensor([0.2, 0.1, 0.0]))
    assert torch.allclose(ego_inverse_ttc, torch.tensor([[0.1]]), atol=1e-6)


def test_block_causal_mask_preserves_same_frame_and_blocks_future() -> None:
    mask = block_causal_attention_mask(steps=3, patches=2)
    assert not mask[0, 1]
    assert mask[0, 2]
    assert not mask[4, 0]
    assert not mask[4, 5]


def test_temporal_mixers_are_causal() -> None:
    torch.manual_seed(3)
    tokens = torch.randn(2, 3, 4, 32)
    changed = tokens.clone()
    changed[:, 2] += 100.0
    for mixer in (
        BlockCausalTransformer(32, heads=4, depth=1).eval(),
        TemporalDeltaMemory(32, heads=4).eval(),
    ):
        first = mixer(tokens)
        second = mixer(changed)
        assert torch.allclose(first[:, :2], second[:, :2], atol=1e-5)


def test_kda_recurrence_matches_one_step_delta_rule() -> None:
    query = torch.tensor([[[[[1.0, 0.0]]]]])
    key = query.clone()
    value = torch.tensor([[[[[2.0, 3.0]]]]])
    retention = torch.ones_like(query)
    beta = torch.tensor([[[[0.25]]]])
    output = kimi_delta_recurrence(query, key, value, retention, beta)
    assert torch.allclose(output, 0.25 * value)


def test_attention_residuals_are_normalized_and_task_specific() -> None:
    router = TaskSpecificAttentionResiduals(32, 3)
    layers = tuple(torch.randn(2, 3, 5, 32) for _ in range(3))
    output = router(layers)
    assert set(output.task_tokens) == {"mask", "motion", "geometry", "risk"}
    for weights in output.task_weights.values():
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))


def test_audited_base_backbone_exposes_dense_intermediate_tokens(tmp_path: Path) -> None:
    encoder = EventTubeletTransformerEncoder(
        21,
        embed_dim=192,
        event_bins=5,
        patch_size=16,
        depth=2,
        num_heads=6,
    )
    checkpoint = tmp_path / "base.pt"
    torch.save(
        {
            "model_name": "event-tubelet-transformer",
            "in_channels": 21,
            "bins": 5,
            "encoder_state_dict": encoder.state_dict(),
        },
        checkpoint,
    )
    backbone = BaseEventTubeletBackbone(checkpoint)
    output = backbone(torch.randn(1, 2, 21, 32, 32))
    assert len(output.layer_tokens) == 2
    assert output.dense_tokens.shape == (1, 2, 20, 192)
    assert output.global_token.shape == (1, 2, 192)


def test_target_query_is_dense_and_differentiable() -> None:
    query = TargetBackgroundQuery(32)
    tokens = torch.randn(2, 12, 32, requires_grad=True)
    output = query(tokens, (3, 4))
    assert output.soft_mask.shape == (2, 3, 4)
    assert output.box_xyxy.shape == (2, 4)
    output.object_token.sum().backward()
    assert tokens.grad is not None


def _inputs() -> dict[str, torch.Tensor]:
    events = torch.randn(2, 3, 4, 16, 24)
    times = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
    boxes = torch.tensor(
        [
            [
                [[0.40, 0.35, 0.58, 0.65]],
                [[0.39, 0.34, 0.60, 0.67]],
                [[0.38, 0.33, 0.62, 0.69]],
            ],
            [
                [[0.30, 0.30, 0.50, 0.60]],
                [[0.29, 0.29, 0.52, 0.62]],
                [[0.28, 0.28, 0.54, 0.64]],
            ],
        ]
    )
    return {
        "context_events": events,
        "context_times_s": times,
        "context_boxes": boxes,
        "context_object_mask": torch.ones(2, 3, 1, dtype=torch.bool),
        "context_ego_actions": torch.randn(2, 3, 8),
        "context_ego_action_mask": torch.ones(2, 3, dtype=torch.bool),
    }


def test_complete_oge_forward_backward_and_diagnostics() -> None:
    model = ObjectGeometryJEPATTC(
        OGEConfig(
            in_channels=4,
            dim=32,
            backbone_depth=3,
            heads=4,
            temporal_depth=1,
            head_mode="dense",
            use_attention_residuals=True,
            use_target_query=True,
            use_highres_refiner=True,
            geometry_mode="router",
            bbox_source="predicted",
            use_yaw_derotation=True,
            use_bounded_residual=True,
            use_uncertainty=True,
        )
    )
    output = model(**_inputs())
    assert output.ttc_seconds.shape == (2,)
    assert output.mask_logits is not None
    assert output.refined_mask_logits is not None
    assert output.geometry_estimates is not None
    assert output.geometry_weights is not None
    assert torch.allclose(
        output.geometry_weights.sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    output.ttc_seconds.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_matched_global_dense_and_attnres_share_common_initialization() -> None:
    common = {
        "in_channels": 4,
        "dim": 32,
        "backbone_depth": 3,
        "heads": 4,
        "temporal_depth": 1,
    }
    models: list[ObjectGeometryJEPATTC] = []
    for overrides in (
        {"head_mode": "global"},
        {"head_mode": "dense"},
        {"head_mode": "dense", "use_attention_residuals": True},
    ):
        torch.manual_seed(17)
        models.append(ObjectGeometryJEPATTC(OGEConfig(**common, **overrides)))

    reference_backbone = models[0].backbone.state_dict()
    reference_head = models[0].direct_log_ttc_head.state_dict()
    for model in models[1:]:
        for name, value in reference_backbone.items():
            assert torch.equal(value, model.backbone.state_dict()[name])
        for name, value in reference_head.items():
            assert torch.equal(value, model.direct_log_ttc_head.state_dict()[name])

    assert models[0].global_temporal is not None
    assert models[1].mixer is not None
    assert models[2].mixer is not None
    global_temporal = models[0].global_temporal.state_dict()
    for dense_model in models[1:]:
        dense_temporal = dense_model.mixer.temporal.state_dict()
        for name, value in global_temporal.items():
            assert torch.equal(value, dense_temporal[name])


def test_matched_global_control_uses_early_causal_frames() -> None:
    torch.manual_seed(19)
    model = ObjectGeometryJEPATTC(
        OGEConfig(
            in_channels=4,
            dim=32,
            backbone_depth=3,
            heads=4,
            temporal_depth=1,
            head_mode="global",
        )
    ).eval()
    inputs = _inputs()
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["context_events"][:, 0] += 3.0
    with torch.inference_mode():
        first = model(**inputs).ttc_seconds
        second = model(**changed).ttc_seconds
    assert not torch.allclose(first, second)


def test_garl_direct_and_height_ratio_arms() -> None:
    events = torch.randn(2, 4, 128, 128)
    rgb_pair = torch.randn(2, 2, 3, 128, 128)
    elapsed = torch.full((2,), 0.2)
    for objective in ("direct", "height_ratio"):
        model = GarlTTCReplica(
            GarlTTCConfig(
                event_channels=4,
                objective=objective,
                dim=32,
                backbone="compact",
                foreground_supervision=True,
            )
        )
        output = model(events, elapsed, rgb_pair=rgb_pair)
        assert output.ttc_seconds.shape == (2,)
        assert output.foreground_logits is not None
        assert output.foreground_logits.shape == (2, 4, 256, 256)
        assert torch.isfinite(output.ttc_seconds).all()
        if objective == "height_ratio":
            assert output.predicted_height_ratio is not None
            assert output.predicted_heights is not None


def test_garl_height_formula_is_exact() -> None:
    from e_jepa_ttc.models.height_ratio_head import LearnedHeightRatioHead

    head = LearnedHeightRatioHead(8)
    with torch.no_grad():
        head.height_regressor.weight.zero_()
        head.height_regressor.bias.copy_(torch.tensor([8.0, 10.0]))
    token = torch.zeros(1, 8)
    inverse, ratio, heights = head(token, torch.tensor([0.1]))
    assert torch.allclose(heights, torch.tensor([[8.0, 10.0]]))
    assert torch.allclose(ratio, torch.tensor([0.8]))
    assert torch.allclose(inverse, torch.tensor([2.0]))


def test_garl_resnet50_parameterization_matches_public_source() -> None:
    direct = GarlTTCReplica(
        GarlTTCConfig(
            event_channels=40,
            modality="event",
            objective="direct",
            foreground_supervision=False,
        )
    )
    lhr = GarlTTCReplica(
        GarlTTCConfig(
            event_channels=40,
            modality="event",
            objective="height_ratio",
            foreground_supervision=False,
        )
    )

    assert direct.event_encoder is not None
    assert direct.event_encoder.conv1.weight.shape == (64, 40, 7, 7)
    assert direct.middle_layer.weight.shape == (512, 2048)
    assert sum(parameter.numel() for parameter in direct.parameters()) == 24_673_665
    assert sum(parameter.numel() for parameter in lhr.parameters()) == 24_674_178


def test_teacher_consensus_is_detached_and_reliability_gated() -> None:
    student = torch.zeros(2, 4, 5, requires_grad=True)
    teachers = torch.randn(2, 3, 4, 5, requires_grad=True)
    reliability = torch.tensor([[1.0, 0.1, 0.0], [0.2, 1.0, 0.5]])
    output = reliability_gated_teacher_target(student, teachers, reliability)
    assert not output.target.requires_grad
    assert torch.allclose(output.weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.weights[:, 2] <= output.weights[:, 1] + 1.0)


def test_roi_curriculum_moves_from_gt_to_predicted() -> None:
    assert predicted_roi_probability(0, 20) == 0.0
    assert predicted_roi_probability(20, 20) == 1.0
    ground_truth = torch.zeros(4, 4)
    predicted = torch.ones(4, 4)
    selected, mask = select_curriculum_boxes(
        ground_truth,
        predicted,
        predicted_probability=1.0,
    )
    assert mask.all()
    assert torch.equal(selected, predicted)


def test_embedding_health_reports_collapsed_dimensions() -> None:
    health = embedding_health(torch.ones(8, 16))
    assert health["collapsed_dimension_fraction"] == 1.0
    assert health["mean_embedding_norm"] > 0.0
