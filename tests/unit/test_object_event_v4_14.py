from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.object_event_v4_13 import ObjectEventV413Config
from e_jepa_ttc.object_event_v4_14 import (
    V414AggregateConfig,
    align_selective_seeds,
    pairwise_seed_metrics,
    robustness_gates,
)


def _frame(probability: list[float], prediction: list[float]) -> pd.DataFrame:
    count = len(probability)
    return pd.DataFrame({
        "sequence_id": ["s"] * count,
        "sample_token": [f"x{i}" for i in range(count)],
        "track_id": ["t"] * count,
        "delta_t_s": [0.1] * count,
        "target_ttc_s": [1.0, -1.0][:count],
        "target_expansion": [0.1, -0.1][:count],
        "baseline_prediction_expansion": [0.08, 0.08][:count],
        "negative_probability": probability,
        "selective_prediction_expansion": prediction,
    })


def test_config_requires_three_unique_seeds() -> None:
    with pytest.raises(ValueError):
        V414AggregateConfig(seeds=(7, 7, 13))


def test_median_probability_requires_majority_for_override() -> None:
    frames = {
        7: _frame([0.99, 0.99], [-0.0016, -0.0016]),
        13: _frame([0.99, 0.20], [-0.0016, 0.048]),
        23: _frame([0.10, 0.10], [0.064, 0.064]),
    }
    aligned = align_selective_seeds(frames, fusion_config=ObjectEventV413Config())
    assert aligned["consensus_override"].tolist() == [True, False]
    assert aligned.loc[0, "consensus_prediction_expansion"] < 0.0
    assert aligned.loc[1, "consensus_prediction_expansion"] > 0.0


def test_alignment_rejects_different_locked_baseline() -> None:
    a = _frame([0.1, 0.2], [0.05, 0.05])
    b = _frame([0.1, 0.2], [0.05, 0.05])
    b.loc[0, "baseline_prediction_expansion"] = 0.09
    with pytest.raises(ValueError, match="baseline differs"):
        align_selective_seeds({7: a, 13: b}, fusion_config=ObjectEventV413Config())


def test_pairwise_metrics_report_probability_and_prediction_agreement() -> None:
    a = _frame([0.1, 0.9], [0.05, -0.01])
    b = _frame([0.2, 0.8], [0.04, -0.02])
    metrics = pairwise_seed_metrics({7: a, 13: b})
    assert metrics.loc[0, "selective_prediction_pearson"] > 0.99
    assert metrics.loc[0, "negative_probability_pearson"] > 0.99


def test_robustness_gates_can_pass() -> None:
    config = V414AggregateConfig(track_bootstrap_repeats=100)
    per_seed = pd.DataFrame({
        "passed": [True, True, False],
        "pearson": [0.68, 0.67, 0.66],
        "negative_accuracy": [0.70, 0.68, 0.66],
        "minimum_sequence_negative_accuracy": [0.40, 0.35, 0.30],
    })
    consensus = {
        "pearson": 0.68,
        "weighted_mid": 190.0,
        "expansion_mae": 0.015,
        "balanced_sign_accuracy": 0.77,
        "negative_accuracy": 0.70,
        "minimum_sequence_pearson": 0.60,
        "minimum_sequence_negative_accuracy": 0.40,
        "positive_accuracy": 0.85,
        "ttc_saturation_rate": 0.08,
        "weighted_rte_percent": 500.0,
    }
    diagnostics = {
        "override_rate": 0.05,
        "mean_sample_prediction_std": 0.004,
        "zero_event_pearson_drop": 0.68,
        "shuffled_event_pearson_drop": 0.70,
    }
    pairwise = pd.DataFrame({
        "selective_prediction_pearson": [0.95, 0.94, 0.96],
        "negative_probability_pearson": [0.70, 0.65, 0.72],
    })
    gates = robustness_gates(
        per_seed=per_seed,
        consensus=consensus,
        baseline={"weighted_rte_percent": 400.0, "ttc_saturation_rate": 0.04},
        diagnostics=diagnostics,
        bootstrap={"lower_95": 0.60},
        pairwise=pairwise,
        config=config,
    )
    assert all(gates.values())


def test_robustness_gates_reject_fragile_negative_sequence() -> None:
    config = V414AggregateConfig(track_bootstrap_repeats=100)
    per_seed = pd.DataFrame({
        "passed": [True, True, True],
        "pearson": [0.68, 0.68, 0.68],
        "negative_accuracy": [0.70, 0.70, 0.70],
        "minimum_sequence_negative_accuracy": [0.40, 0.40, 0.40],
    })
    consensus = {
        "pearson": 0.68,
        "weighted_mid": 190.0,
        "expansion_mae": 0.015,
        "balanced_sign_accuracy": 0.77,
        "negative_accuracy": 0.70,
        "minimum_sequence_pearson": 0.60,
        "minimum_sequence_negative_accuracy": 0.10,
        "positive_accuracy": 0.85,
        "ttc_saturation_rate": 0.08,
        "weighted_rte_percent": 500.0,
    }
    diagnostics = {
        "override_rate": 0.05,
        "mean_sample_prediction_std": 0.004,
        "zero_event_pearson_drop": 0.68,
        "shuffled_event_pearson_drop": 0.70,
    }
    pairwise = pd.DataFrame({
        "selective_prediction_pearson": [0.95],
        "negative_probability_pearson": [0.70],
    })
    gates = robustness_gates(
        per_seed=per_seed,
        consensus=consensus,
        baseline={"weighted_rte_percent": 400.0, "ttc_saturation_rate": 0.04},
        diagnostics=diagnostics,
        bootstrap={"lower_95": 0.60},
        pairwise=pairwise,
        config=config,
    )
    assert not gates["consensus_min_sequence_negative_accuracy"]
