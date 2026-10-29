from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_17 import ObjectEventTTCV417, ObjectEventV417Config
from e_jepa_ttc.training.object_event_v4_17 import (
    ObjectEventV417LossConfig,
    signed_anchor_features,
    signed_anchor_logits,
    temporal_sign_loss,
    v417_screen_gates,
)


def test_signed_anchor_features_and_logits_are_odd() -> None:
    anchor = torch.tensor([-0.02, -0.005, 0.0, 0.005, 0.02])
    a = signed_anchor_features(anchor, train_scale=0.01, clip=6.0)
    b = signed_anchor_features(-anchor, train_scale=0.01, clip=6.0)
    la = signed_anchor_logits(anchor, train_scale=0.01, clip=6.0, strength=1.0)
    lb = signed_anchor_logits(-anchor, train_scale=0.01, clip=6.0, strength=1.0)
    assert torch.allclose(a, -b, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(la, -lb, atol=1.0e-7, rtol=0.0)
    assert la[0] > 0.0  # negative expansion -> negative-class logit
    assert la[-1] < 0.0


def test_v417_anchor_residual_sign_is_exactly_odd_and_magnitude_frozen() -> None:
    torch.manual_seed(17)
    config = ObjectEventV417Config(window_size=4)
    model = ObjectEventTTCV417(12, config).eval()
    windows = torch.randn(6, 4, 12)
    mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    magnitude = torch.tensor([0.003, 0.005, 0.01, 0.02, 0.04, 0.08])
    anchor_logits = torch.randn(6, 4)
    anchor_logits = anchor_logits.masked_fill(~mask, 0.0)
    with torch.no_grad():
        a = model(windows, mask, magnitude, anchor_logits)
        b = model(-windows, mask, magnitude, -anchor_logits)
    assert torch.max(torch.abs(a.sign_logit + b.sign_logit)).item() < 1.0e-6
    assert torch.allclose(a.magnitude, magnitude, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(b.magnitude, magnitude, atol=1.0e-7, rtol=0.0)


def test_v417_residual_is_bounded() -> None:
    torch.manual_seed(18)
    config = ObjectEventV417Config(window_size=3, maximum_residual_logit=1.5)
    model = ObjectEventTTCV417(8, config).eval()
    windows = 1000.0 * torch.randn(10, 3, 8)
    mask = torch.ones(10, 3, dtype=torch.bool)
    magnitude = torch.full((10,), 0.02)
    anchor_logits = torch.zeros(10, 3)
    with torch.no_grad():
        output = model(windows, mask, magnitude, anchor_logits)
    assert torch.max(torch.abs(output.residual_logit)).item() <= 1.5 + 1.0e-6


def test_strong_anchor_cannot_be_flipped_by_bounded_residual() -> None:
    torch.manual_seed(19)
    config = ObjectEventV417Config(window_size=3, maximum_residual_logit=1.5)
    model = ObjectEventTTCV417(8, config).eval()
    windows = torch.randn(5, 3, 8)
    mask = torch.ones(5, 3, dtype=torch.bool)
    magnitude = torch.full((5,), 0.02)
    # Strong positive expansion anchor -> strongly negative negative-class logit.
    anchor_logits = torch.full((5, 3), -2.0)
    with torch.no_grad():
        output = model(windows, mask, magnitude, anchor_logits)
    assert torch.all(output.sign_logit < 0.0)
    assert torch.all(output.signed_expansion > 0.0)


def test_v417_sign_loss_is_finite() -> None:
    batch, steps = 8, 4
    sign = torch.randn(batch, requires_grad=True)
    instant = torch.randn(batch, steps, requires_grad=True)
    target = torch.tensor([0.02, -0.03, 0.01, -0.01, 0.04, -0.02, 0.03, -0.05])
    history = target[:, None].repeat(1, steps)
    mask = torch.ones(batch, steps, dtype=torch.bool)
    loss = temporal_sign_loss(
        sign_logit=sign,
        instant_sign_logits=instant,
        target_expansion=target,
        history_target_expansion=history,
        history_mask=mask,
        config=ObjectEventV417LossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert sign.grad is not None


def test_v417_gates_cover_anchor_and_temporal_contract() -> None:
    oof = {
        "balanced_sign_accuracy": 0.84,
        "negative_accuracy": 0.76,
        "minimum_sequence_negative_accuracy": 0.55,
    }
    validation = {
        "pearson": 0.68,
        "expansion_mae": 0.015,
        "weighted_mid": 190.0,
        "weighted_rte_percent": 500.0,
        "ttc_saturation_rate": 0.06,
        "balanced_sign_accuracy": 0.79,
        "negative_accuracy": 0.71,
        "minimum_sequence_pearson": 0.59,
        "minimum_sequence_negative_accuracy": 0.45,
        "positive_accuracy": 0.86,
    }
    baseline = {"weighted_rte_percent": 433.0, "ttc_saturation_rate": 0.056}
    anchor = {"magnitude_mae": 0.0125}
    diagnostics = {
        "zero_event_pearson_drop": 0.6,
        "shuffled_event_pearson_drop": 0.6,
        "sign_oddness_max_abs": 0.0,
    }
    gates = {
        "oof_balanced_sign_gate": 0.78,
        "oof_negative_accuracy_gate": 0.70,
        "oof_min_sequence_negative_accuracy_gate": 0.40,
        "validation_pearson_gate": 0.65,
        "validation_expansion_mae_gate": 0.0165,
        "validation_weighted_mid_gate": 210.0,
        "validation_weighted_rte_relative_ceiling": 1.35,
        "validation_saturation_max_increase": 0.02,
        "validation_balanced_sign_gate": 0.77,
        "validation_negative_accuracy_gate": 0.68,
        "validation_min_sequence_pearson_gate": 0.57,
        "validation_min_sequence_negative_accuracy_gate": 0.40,
        "validation_positive_accuracy_gate": 0.84,
        "anchor_magnitude_mae_gate": 0.0130,
        "zero_event_pearson_drop_gate": 0.40,
        "shuffled_event_pearson_drop_gate": 0.40,
        "exact_sign_oddness_ceiling": 1.0e-5,
    }
    result = v417_screen_gates(
        oof=oof,
        validation=validation,
        baseline=baseline,
        anchor=anchor,
        diagnostics=diagnostics,
        gates=gates,
    )
    assert all(result.values())
