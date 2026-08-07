from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import pytest
from torch import nn

from e_jepa_ttc.models.object_event_v4_15 import (
    ObjectEventTTCV415,
    ObjectEventV415Config,
    sign_magnitude_projection,
)
from e_jepa_ttc.training.object_event_v4_15 import (
    positive_magnitude_ceiling,
    positive_tail_threshold,
    projected_numpy,
    select_oof_threshold,
    v415_screen_gates,
)


class _DummyExtractor(nn.Module):
    descriptor_dim = 4

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def _descriptor_and_base(self, events: torch.Tensor):
        values = events.flatten(1)[:, :4] * self.scale
        return values, values[:, 0]


def test_shared_descriptor_and_head_are_odd() -> None:
    model = ObjectEventTTCV415(
        [_DummyExtractor(1.0), _DummyExtractor(2.0), _DummyExtractor(0.5)],
        ObjectEventV415Config(hidden_dim=8, bottleneck_dim=4),
    )
    events = torch.randn(5, 1, 1, 2, 2)
    descriptor = model.consensus_descriptor(events)
    reversed_descriptor = model.consensus_descriptor(-events)
    assert torch.allclose(reversed_descriptor, -descriptor, atol=1.0e-6)
    logits = model.sign_head(descriptor)
    reversed_logits = model.sign_head(reversed_descriptor)
    assert torch.allclose(reversed_logits, -logits, atol=1.0e-6)


def test_sign_magnitude_projection_never_changes_magnitude() -> None:
    baseline = torch.tensor([0.2, 0.1, -0.3])
    probability = torch.tensor([0.99, 0.2, 0.01])
    result = sign_magnitude_projection(
        baseline, probability, negative_threshold=0.95
    )
    assert torch.allclose(result.abs(), baseline.abs())
    assert result.tolist() == pytest.approx([-0.2, 0.1, -0.3])


def test_numpy_projection_only_flips_positive_baseline() -> None:
    baseline = np.asarray([0.2, -0.1, 0.3])
    probability = np.asarray([0.9, 0.99, 0.2])
    prediction, override = projected_numpy(baseline, probability, 0.8)
    assert np.allclose(prediction, [-0.2, -0.1, 0.3])
    assert override.tolist() == [True, False, False]


def test_magnitude_ceiling_uses_empirical_train_quantile() -> None:
    magnitude = np.asarray([0.01, 0.02, 0.03, 0.04])
    assert positive_magnitude_ceiling(magnitude, 0.25) == pytest.approx(0.02)


def test_projection_can_abstain_on_large_positive_magnitude() -> None:
    baseline = np.asarray([0.01, 0.04, -0.02])
    probability = np.asarray([0.99, 0.99, 0.99])
    prediction, override = projected_numpy(
        baseline, probability, 0.95, magnitude_ceiling=0.02
    )
    assert np.allclose(prediction, [-0.01, 0.04, -0.02])
    assert override.tolist() == [True, False, False]


def test_positive_tail_threshold_uses_empirical_upper_tail() -> None:
    probability = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    assert positive_tail_threshold(probability, 0.2) == pytest.approx(0.5)


def test_oof_threshold_selection_uses_direct_sign_tail_budget() -> None:
    positive_probability = np.linspace(0.1, 0.7, 100)
    negative_probability = np.linspace(0.68, 0.95, 40)
    probability = np.concatenate((negative_probability, positive_probability))
    target = np.concatenate((-np.ones(40), np.ones(100)))
    frame = pd.DataFrame(
        {
            "sequence_id": ["a"] * 70 + ["b"] * 70,
            "target_expansion": target,
            "fused_prediction_expansion": np.ones(140),
        }
    )
    selected, sweep = select_oof_threshold(
        frame,
        probability,
        tail_alphas=[0.01, 0.02, 0.03],
        positive_accuracy_floor=0.95,
        override_rate_ceiling=0.40,
        minimum_negatives=10,
    )
    assert selected.feasible
    assert selected.tail_alpha in {0.01, 0.02, 0.03}
    assert selected.threshold <= float(positive_probability.max())
    assert selected.negative_accuracy > 0.0
    assert {"tail_alpha", "threshold"}.issubset(sweep.columns)


def test_screen_gates_include_calibration_and_sequence_sign() -> None:
    oof = {
        "balanced_sign_accuracy": 0.8,
        "negative_accuracy": 0.7,
        "minimum_sequence_negative_accuracy": 0.4,
    }
    validation = {
        "pearson": 0.7,
        "expansion_mae": 0.015,
        "weighted_mid": 190.0,
        "weighted_rte_percent": 500.0,
        "ttc_saturation_rate": 0.06,
        "balanced_sign_accuracy": 0.78,
        "negative_accuracy": 0.7,
        "minimum_sequence_pearson": 0.6,
        "minimum_sequence_negative_accuracy": 0.4,
        "positive_accuracy": 0.86,
    }
    baseline = {
        "weighted_rte_percent": 433.0,
        "ttc_saturation_rate": 0.056,
    }
    diagnostics = {
        "override_rate": 0.05,
        "zero_event_pearson_drop": 0.7,
        "shuffled_event_pearson_drop": 0.7,
    }
    gates_cfg = {
        "oof_balanced_sign_gate": 0.75,
        "oof_negative_accuracy_gate": 0.65,
        "oof_min_sequence_negative_accuracy_gate": 0.25,
        "validation_pearson_gate": 0.67,
        "validation_expansion_mae_gate": 0.0157,
        "validation_weighted_mid_gate": 200.0,
        "validation_weighted_rte_relative_ceiling": 1.5,
        "validation_saturation_max_increase": 0.04,
        "validation_balanced_sign_gate": 0.77,
        "validation_negative_accuracy_gate": 0.68,
        "validation_min_sequence_pearson_gate": 0.57,
        "validation_min_sequence_negative_accuracy_gate": 0.35,
        "validation_positive_accuracy_gate": 0.84,
        "validation_override_rate_ceiling": 0.08,
        "zero_event_pearson_drop_gate": 0.55,
        "shuffled_event_pearson_drop_gate": 0.55,
    }
    assert all(
        v415_screen_gates(
            oof=oof,
            validation=validation,
            baseline=baseline,
            diagnostics=diagnostics,
            gates=gates_cfg,
        ).values()
    )
