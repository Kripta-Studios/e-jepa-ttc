from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.object_event_v4_16 import ObjectEventTTCV416, ObjectEventV416Config
from e_jepa_ttc.training.object_event_v4_16 import (
    ObjectEventV416LossConfig,
    causal_window_indices,
    gather_scalar_windows,
    gather_windows,
    temporal_dual_head_loss,
    uniform_epoch_indices,
    v416_screen_gates,
)


def test_causal_windows_sort_within_track_and_right_align() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s", "s", "s", "s"],
            "track_id": ["a", "a", "b", "a"],
            "sample_token": ["s_a_300", "s_a_100", "s_b_200", "s_a_200"],
        }
    )
    index, mask, length = causal_window_indices(frame, window_size=3)
    assert index[0].tolist() == [1, 3, 0]
    assert index[1].tolist() == [-1, -1, 1]
    assert index[3].tolist() == [-1, 1, 3]
    assert mask[2].tolist() == [False, False, True]
    assert length.tolist() == [3, 1, 1, 2]


def test_gather_windows_zero_pads_history() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "track_id": ["a", "a"],
            "sample_token": ["s_a_100", "s_a_200"],
        }
    )
    index, mask, _ = causal_window_indices(frame, window_size=3)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    windows = gather_windows(values, index, mask)
    scalar = gather_scalar_windows(torch.tensor([5.0, 6.0]), index, mask)
    assert windows.shape == (2, 3, 2)
    assert torch.equal(windows[0, :2], torch.zeros(2, 2))
    assert torch.equal(windows[1, -2], values[0])
    assert scalar[1].tolist() == [0.0, 5.0, 6.0]


def test_temporal_sign_is_odd_and_magnitude_is_even() -> None:
    torch.manual_seed(7)
    config = ObjectEventV416Config(window_size=4, magnitude_floor=1.0e-4)
    model = ObjectEventTTCV416(10, config).eval()
    windows = torch.randn(5, 4, 10)
    mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    anchor = torch.full((5,), 0.02)
    with torch.no_grad():
        a = model(windows, mask, anchor)
        b = model(-windows, mask, anchor)
    assert torch.max(torch.abs(a.sign_logit + b.sign_logit)).item() < 1.0e-6
    assert torch.max(torch.abs(a.magnitude - b.magnitude)).item() < 1.0e-7
    assert torch.all(a.magnitude > 0.0)


def test_zero_initialised_magnitude_residual_preserves_anchor() -> None:
    config = ObjectEventV416Config(window_size=3)
    model = ObjectEventTTCV416(8, config).eval()
    windows = torch.randn(4, 3, 8)
    mask = torch.ones(4, 3, dtype=torch.bool)
    anchor = torch.tensor([0.005, 0.01, 0.02, 0.04])
    with torch.no_grad():
        output = model(windows, mask, anchor)
    assert torch.allclose(output.magnitude, anchor, atol=1.0e-7, rtol=0.0)


def test_temporal_dual_head_loss_is_finite() -> None:
    batch, steps = 6, 4
    sign_logit = torch.randn(batch, requires_grad=True)
    instant = torch.randn(batch, steps, requires_grad=True)
    magnitude = torch.rand(batch, requires_grad=True) * 0.05 + 0.001
    target = torch.tensor([0.02, -0.03, 0.01, -0.01, 0.04, -0.02])
    teacher = target.abs() * 0.9
    history_target = target[:, None].repeat(1, steps)
    mask = torch.ones(batch, steps, dtype=torch.bool)
    weight = torch.ones(batch)
    loss = temporal_dual_head_loss(
        sign_logit=sign_logit,
        instant_sign_logits=instant,
        magnitude=magnitude,
        target_expansion=target,
        teacher_magnitude=teacher,
        history_target_expansion=history_target,
        history_mask=mask,
        sample_weight=weight,
        magnitude_floor=1.0e-4,
        config=ObjectEventV416LossConfig(),
    )
    assert torch.isfinite(loss.total)
    assert float(loss.total.detach()) > 0.0


def test_v416_screen_gates_cover_temporal_contract() -> None:
    oof = {
        "balanced_sign_accuracy": 0.80,
        "negative_accuracy": 0.70,
        "minimum_sequence_negative_accuracy": 0.45,
        "expansion_mae": 0.015,
    }
    baseline = {"weighted_rte_percent": 400.0, "ttc_saturation_rate": 0.05}
    validation = {
        "pearson": 0.68,
        "expansion_mae": 0.015,
        "weighted_mid": 190.0,
        "weighted_rte_percent": 430.0,
        "ttc_saturation_rate": 0.05,
        "balanced_sign_accuracy": 0.79,
        "negative_accuracy": 0.72,
        "minimum_sequence_pearson": 0.58,
        "minimum_sequence_negative_accuracy": 0.43,
        "positive_accuracy": 0.86,
    }
    diagnostics = {
        "zero_event_pearson_drop": 0.60,
        "shuffled_event_pearson_drop": 0.60,
        "sign_oddness_max_abs": 1.0e-7,
        "magnitude_evenness_max_abs": 0.0,
    }
    gates = {
        "oof_balanced_sign_gate": 0.75,
        "oof_negative_accuracy_gate": 0.65,
        "oof_min_sequence_negative_accuracy_gate": 0.30,
        "oof_expansion_mae_gate": 0.018,
        "validation_pearson_gate": 0.65,
        "validation_expansion_mae_gate": 0.0175,
        "validation_weighted_mid_gate": 220.0,
        "validation_weighted_rte_relative_ceiling": 1.5,
        "validation_saturation_max_increase": 0.05,
        "validation_balanced_sign_gate": 0.77,
        "validation_negative_accuracy_gate": 0.68,
        "validation_min_sequence_pearson_gate": 0.57,
        "validation_min_sequence_negative_accuracy_gate": 0.40,
        "validation_positive_accuracy_gate": 0.84,
        "zero_event_pearson_drop_gate": 0.40,
        "shuffled_event_pearson_drop_gate": 0.40,
        "exact_sign_oddness_ceiling": 1.0e-5,
        "exact_magnitude_evenness_ceiling": 1.0e-6,
    }
    result = v416_screen_gates(
        oof=oof,
        validation=validation,
        baseline=baseline,
        diagnostics=diagnostics,
        gates=gates,
    )
    assert result and all(result.values())


def test_uniform_epoch_indices_visits_each_row_once() -> None:
    indices = torch.tensor([9, 3, 7, 1, 5], dtype=torch.long)
    generator = torch.Generator().manual_seed(123)
    sampled = uniform_epoch_indices(indices, generator)
    assert sorted(sampled.tolist()) == sorted(indices.tolist())
    assert len(torch.unique(sampled)) == len(indices)


def test_uniform_epoch_indices_rejects_empty() -> None:
    generator = torch.Generator().manual_seed(1)
    try:
        uniform_epoch_indices(torch.empty(0, dtype=torch.long), generator)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty training indices must fail")
