from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.data.evttc_garl_adapter import (
    make_garl_model_input,
    reject_labels_from_predict_payload,
)
from e_jepa_ttc.evaluation.scientific_gates import highres_kda_gate
from e_jepa_ttc.evaluation.token_complexity import (
    global_attention_pairs,
    patch_grid_tokens,
    temporal_factorized_pairs,
)
from e_jepa_ttc.losses.garl_ttc import signed_log_ttc_loss
from e_jepa_ttc.losses.jepa_dense import cosine_prediction_loss
from e_jepa_ttc.models.highres_token_pyramid import SpaceToDepthPatchMerge


def test_evttc_adapter_validates_model_only_input() -> None:
    model_input = make_garl_model_input(
        torch.zeros(2, 2, 20, 128, 128),
        torch.zeros(2, 2, dtype=torch.int64),
        torch.full((2,), 0.1),
    )
    assert model_input.protocol_id == "evttc_garl_p0_zero_shot_v1"
    with pytest.raises(ValueError, match="forbidden labels"):
        reject_labels_from_predict_payload({"ttc_s": 1.0})


def test_space_to_depth_and_mask_preserve_spatial_contract() -> None:
    tokens = torch.arange(1 * 2 * 3 * 5 * 2, dtype=torch.float32).reshape(1, 2, 3, 5, 2)
    valid = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    valid[:, :, -1, -1] = False
    merge = SpaceToDepthPatchMerge(2, 4)
    output, output_mask = merge(tokens, valid)
    assert output.shape == (1, 2, 2, 3, 4)
    assert output_mask.shape == (1, 2, 2, 3)
    assert output_mask[..., 0, 0].all()
    assert not output_mask[..., -1, -1].any()


def test_scientific_gate_rejects_observed_kda_regression() -> None:
    decision = highres_kda_gate(
        control_mid=10.0,
        candidate_mid=10.1,
        control_rte_pct=20.0,
        candidate_rte_pct=21.0,
        control_latency_ms=4.0,
        candidate_latency_ms=8.0,
        control_failure_rate_pct=0.0,
        candidate_failure_rate_pct=0.0,
        control_small_object_error=10.0,
        candidate_small_object_error=10.0,
        control_peak_vram_bytes=10_000,
        candidate_peak_vram_bytes=10_000,
        control_max_temporal_steps=5,
        candidate_max_temporal_steps=5,
        temporal_lengths_evaluated=(2, 5),
        temporal_lengths_with_advantage=(),
        seeds=(7, 13, 23),
        seeds_with_advantage=(7, 13),
        config_equivalent=True,
    )
    assert decision.passed is False
    assert "RTE_regression" in decision.reasons


def test_loss_and_complexity_helpers_are_finite() -> None:
    values = torch.randn(4, 8)
    assert torch.isfinite(cosine_prediction_loss(values, values)).item()
    assert torch.isfinite(signed_log_ttc_loss(values, values)).item()
    assert patch_grid_tokens(192, 320, 8) == 960
    assert temporal_factorized_pairs(5, 960) < global_attention_pairs(5, 960)
