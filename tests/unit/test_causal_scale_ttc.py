from __future__ import annotations

import inspect
import math

import pytest
import torch

from e_jepa_ttc.evaluation.causal_scale_v5 import (
    evaluate_operator_gates,
    synthetic_operator_metrics,
)
from e_jepa_ttc.losses.causal_scale_ttc import causal_scale_ttc_loss
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    blend_current_inverse_ttc,
    log_ratio_to_inverse_ttc,
    smooth_temporal_foreground_logits,
    soft_vertical_extent_from_logits,
    target_log_ratio_from_ttc,
)
from scripts.evaluate_causal_scale_v5_operator import _classify_worktree_status


def _rectangle_logits(
    height: int,
    *,
    canvas: int = 96,
    center_y: int = 48,
    width: int = 40,
) -> torch.Tensor:
    logits = torch.full((1, 1, canvas, canvas), -20.0)
    top = center_y - height // 2
    left = (canvas - width) // 2
    logits[..., top : top + height, left : left + width] = 20.0
    return logits


def _small_model() -> CausalScaleTTC:
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
        )
    )


def test_scale_identity_recovers_signed_ttc_and_invalid_post_contact() -> None:
    ttc = torch.tensor([2.0, -2.0, -0.05])
    delta = torch.tensor([0.1, 0.1, 0.1])
    ratio, valid = target_log_ratio_from_ttc(ttc, delta)
    assert valid.tolist() == [True, True, False]
    inverse = log_ratio_to_inverse_ttc(ratio[:2], delta[:2])
    assert torch.allclose(inverse, ttc[:2].reciprocal(), atol=1.0e-6)
    assert ratio[0] > 0.0
    assert ratio[1] < 0.0


def test_soft_extent_is_scale_equivariant_and_translation_invariant() -> None:
    pairs = ((40, 32), (36, 32), (32, 36), (32, 40))
    target: list[torch.Tensor] = []
    predicted: list[torch.Tensor] = []
    for previous_height, current_height in pairs:
        previous = soft_vertical_extent_from_logits(_rectangle_logits(previous_height))
        current = soft_vertical_extent_from_logits(_rectangle_logits(current_height))
        target.append(torch.tensor(math.log(current_height / previous_height)))
        predicted.append(current.height_normalized.log() - previous.height_normalized.log())
    target_tensor = torch.stack(target)
    predicted_tensor = torch.cat(predicted)
    slope = torch.dot(target_tensor, predicted_tensor) / torch.dot(target_tensor, target_tensor)
    correlation = torch.corrcoef(torch.stack((target_tensor, predicted_tensor)))[0, 1]
    assert correlation.item() > 0.999
    assert slope.item() == pytest.approx(1.0, abs=1.0e-4)
    assert torch.equal(torch.sign(target_tensor), torch.sign(predicted_tensor))

    shifted = soft_vertical_extent_from_logits(_rectangle_logits(32, center_y=60))
    centered = soft_vertical_extent_from_logits(_rectangle_logits(32, center_y=44))
    assert shifted.height_normalized.item() == pytest.approx(
        centered.height_normalized.item(),
        abs=1.0e-6,
    )


def test_learned_residual_is_exactly_antisymmetric() -> None:
    torch.manual_seed(4)
    model = _small_model().eval()
    previous = torch.randn(5, 24)
    current = torch.randn(5, 24)
    with torch.inference_mode():
        forward = model.residual(previous, current)
        reverse = model.residual(current, previous)
    assert torch.equal(forward, -reverse)
    assert bool((forward.abs() <= model.config.max_abs_log_ratio_residual).all())


def test_event_model_has_no_box_input_and_marks_zero_events_unknown() -> None:
    signature = inspect.signature(CausalScaleTTC.forward)
    assert tuple(signature.parameters) == ("self", "inputs", "delta_t_s")
    model = _small_model().eval()
    inputs = torch.zeros(2, 3, 2, 32, 32)
    delta = torch.full((2, 2), 0.1)
    with torch.inference_mode():
        output = model(inputs, delta)
    assert output.ttc_mean_seconds.shape == (2,)
    assert output.collision_logits.shape == (2, 4)
    assert output.log_height_ratio.shape == (2, 2)
    assert output.foreground_logits.shape == (2, 3, 1, 32, 32)
    assert output.geometry_tokens.shape == (2, 3, 24)
    assert output.pair_tokens.shape == (2, 2, 24)
    assert output.known_mask.tolist() == [False, False]
    assert torch.equal(output.sensor_support, torch.zeros_like(output.sensor_support))
    assert torch.equal(output.collision_logits, torch.zeros_like(output.collision_logits))
    assert torch.isfinite(output.ttc_mean_seconds).all()


def test_causal_scale_loss_is_finite_and_reaches_foreground_and_auxiliary_heads() -> None:
    torch.manual_seed(9)
    model = _small_model().train()
    inputs = torch.randn(2, 3, 2, 32, 32)
    delta = torch.full((2, 2), 0.1)
    output = model(inputs, delta)
    target_masks = torch.zeros(2, 3, 1, 32, 32)
    target_masks[..., 8:24, 10:22] = 1.0
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=torch.tensor([2.0, -2.0]),
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_masks=target_masks,
        mask_valid=torch.ones(2, 3, dtype=torch.bool),
    )
    assert torch.isfinite(result.total)
    assert result.counts["physical_ratio"] == 2
    assert result.counts["supervised_ttc"] == 2
    assert result.counts["foreground"] == 6
    assert result.counts["temporal_consistency"] in {0, 2}
    result.total.backward()
    assert model.encoder.foreground.weight.grad is not None
    auxiliary = model.auxiliary_inverse_ttc_head[-1]
    assert isinstance(auxiliary, torch.nn.Linear)
    assert auxiliary.weight.grad is not None


def test_configuration_rejects_unsorted_risk_thresholds() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CausalScaleTTCConfig(risk_thresholds_s=(1.0, 0.5))


def test_temporal_inverse_ttc_blend_transports_previous_pair_to_current_time() -> None:
    pair_inverse = torch.tensor([[0.4, 0.5], [0.4, 0.5]])
    pair_known = torch.tensor([[True, True], [False, True]])
    delta = torch.tensor([0.1, 0.1])

    blended, used = blend_current_inverse_ttc(
        pair_inverse,
        pair_known,
        delta,
        current_pair_weight=0.75,
    )

    transported = 0.4 / (1.0 - 0.1 * 0.4)
    assert blended[0] == pytest.approx(0.75 * 0.5 + 0.25 * transported)
    assert blended[1] == pytest.approx(0.5)
    assert used.tolist() == [True, False]


def test_temporal_foreground_consensus_is_reversal_equivariant() -> None:
    logits = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1, 1)

    smoothed = smooth_temporal_foreground_logits(logits, neighbor_weight=0.15)
    reversed_smoothed = smooth_temporal_foreground_logits(
        logits.flip(1), neighbor_weight=0.15
    ).flip(1)

    assert torch.allclose(smoothed, reversed_smoothed, atol=1.0e-6, rtol=0.0)
    assert smoothed[0, 0, 0, 0, 0] == pytest.approx(0.15)
    assert smoothed[0, 2, 0, 0, 0] == pytest.approx(2.0)
    assert smoothed[0, -1, 0, 0, 0] == pytest.approx(3.85)
    assert smooth_temporal_foreground_logits(logits, neighbor_weight=0.0) is logits


def test_temporal_foreground_consensus_rejects_unsafe_weight() -> None:
    with pytest.raises(ValueError, match="neighbor_weight"):
        smooth_temporal_foreground_logits(
            torch.zeros(1, 3, 1, 2, 2), neighbor_weight=0.41
        )


def test_stride_free_foreground_path_has_small_integer_translation_leakage() -> None:
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
            foreground_decoder="equivariant_separable",
            foreground_fullres_dim=8,
        )
    ).eval()
    inputs = torch.zeros(2, 3, 12, 64, 64)
    for endpoint, (top, bottom) in enumerate(((20, 40), (19, 41), (18, 42))):
        inputs[:, endpoint, :, top:bottom, 20:44] = 1.0
    shifted = torch.roll(inputs, shifts=(5, -4), dims=(-2, -1))
    delta = torch.full((2, 2), 0.1)

    with torch.inference_mode():
        reference = model(inputs, delta)
        translated = model(shifted, delta)

    leakage = (reference.log_height_ratio - translated.log_height_ratio).abs().max()
    assert float(leakage) < 1.0e-3


def test_synthetic_operator_protocol_passes_only_mechanistic_gates() -> None:
    config = CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=16,
        geometry_dim=24,
        residual_depth=1,
        dropout=0.0,
    )
    metrics = synthetic_operator_metrics(config, seed=7)
    gates = evaluate_operator_gates(metrics)
    assert gates["passed"] is True
    assert metrics["analytic_pearson"] > 0.999
    assert metrics["slope"] == pytest.approx(1.0, abs=1.0e-4)
    assert metrics["zero_unknown"] == 1.0
    assert metrics["parameter_count"] < 5_000_000


def test_synthetic_operator_gate_fails_closed_on_missing_metric() -> None:
    assert evaluate_operator_gates({"analytic_pearson": 1.0}) == {
        "finite": False,
        "passed": False,
    }


def test_clean_gate_ignores_root_handoffs_but_rejects_untracked_code() -> None:
    root_only = _classify_worktree_status(["?? historical.patch", "?? results.zip"])
    assert root_only == {
        "tracked_dirty": False,
        "untracked_file_count": 2,
        "untracked_code_paths": [],
    }
    code = _classify_worktree_status(["?? src/e_jepa_ttc/new_model.py"])
    assert code["untracked_code_paths"] == ["src/e_jepa_ttc/new_model.py"]
    tracked = _classify_worktree_status([" M README.md"])
    assert tracked["tracked_dirty"] is True
