from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from e_jepa_ttc.object_event_v4_11 import (
    FEATURE_NAMES,
    V411RouterConfig,
    align_seed_experts,
    apply_negative_repair,
    fit_weighted_logistic,
    router_features,
    select_router_train_only,
    validation_gates,
)


def _v49_frame(*, seed_offset: float = 0.0, sequences: int = 3, rows_per_sequence: int = 24) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for sequence_index in range(sequences):
        sequence = f"seq-{sequence_index}"
        for local in range(rows_per_sequence):
            negative = local < rows_per_sequence // 3
            magnitude = 0.01 + 0.001 * local
            target = -magnitude if negative else magnitude
            base = target * 0.75 + seed_offset + (0.003 if negative else -0.001)
            dense = target * 0.90 - seed_offset + (0.001 if negative else 0.001)
            fixed = 0.5 * (base + dense)
            rows.append(
                {
                    "sequence_id": sequence,
                    "sample_token": f"sample-{index:04d}",
                    "track_id": f"track-{sequence_index}-{local // 6}",
                    "delta_t_s": 0.05,
                    "target_ttc_s": 0.05 / target,
                    "target_expansion": target,
                    "base_prediction_expansion": base,
                    "dense_prediction_expansion": dense,
                    "base_zero_events_expansion": 0.002,
                    "dense_zero_events_expansion": 0.0,
                    "base_shuffled_mean_expansion": -base * 0.1,
                    "dense_shuffled_mean_expansion": -dense * 0.1,
                    "fused_prediction_expansion": fixed,
                    "fused_zero_events_expansion": 0.001,
                    "fused_shuffled_mean_expansion": -fixed * 0.1,
                    "expert_disagreement": abs(base - dense),
                }
            )
            index += 1
    return pd.DataFrame.from_records(rows)


def _aligned() -> pd.DataFrame:
    frames = {
        7: _v49_frame(seed_offset=-0.001),
        13: _v49_frame(seed_offset=0.0),
        23: _v49_frame(seed_offset=0.001),
    }
    return align_seed_experts(frames, split_name="test", seeds=(7, 13, 23))


def test_align_and_features_are_seed_invariant() -> None:
    aligned = _aligned()
    features, fixed = router_features(aligned, branch="event", seeds=(7, 13, 23))
    reverse_features, reverse_fixed = router_features(
        aligned, branch="event", seeds=(23, 13, 7)
    )
    assert features.shape == (len(aligned), len(FEATURE_NAMES))
    assert np.allclose(features, reverse_features)
    assert np.allclose(fixed, reverse_fixed)


def test_weighted_logistic_learns_binary_signal() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    features = np.column_stack((x, x**2))
    negative = (x < 0.0).astype(np.float64)
    model = fit_weighted_logistic(features, negative, l2=0.01)
    probability = model.predict_negative_probability(features)
    assert np.mean(probability[x < -1.0]) > 0.9
    assert np.mean(probability[x > 1.0]) < 0.1


def test_negative_repair_never_changes_existing_negatives() -> None:
    fixed = np.asarray([-0.03, 0.02, 0.04])
    probability = np.asarray([0.99, 0.80, 0.20])
    routed, repaired = apply_negative_repair(
        fixed,
        probability,
        threshold=0.6,
        flip_scale=0.5,
        minimum_flip_magnitude=0.002,
    )
    assert routed[0] == fixed[0]
    assert routed[1] < 0.0
    assert routed[2] == fixed[2]
    assert repaired.tolist() == [False, True, False]


def test_train_only_selection_returns_grouped_oof_predictions() -> None:
    aligned = _aligned()
    config = V411RouterConfig(
        seeds=(7, 13, 23),
        l2_grid=(0.01, 0.1),
        negative_threshold_grid=(0.4, 0.6),
        flip_scale_grid=(0.5, 1.0),
        per_sequence_negative_min_count=2,
        cv_pearson_floor=0.5,
        cv_expansion_mae_ceiling=0.05,
        track_bootstrap_repeats=100,
    )
    model, selected, candidates, oof = select_router_train_only(aligned, config=config)
    assert selected["l2"] in config.l2_grid
    assert len(candidates) == 8
    assert len(oof) == len(aligned)
    assert np.isfinite(oof["negative_probability"]).all()
    assert model.feature_names == FEATURE_NAMES


def test_validation_gates_keep_negative_sequence_gate_independent() -> None:
    config = V411RouterConfig(track_bootstrap_repeats=100)
    baseline = {
        "pearson": 0.67,
        "expansion_mae": 0.015,
    }
    routed = {
        "pearson": 0.66,
        "expansion_mae": 0.0155,
        "balanced_sign_accuracy": 0.76,
        "negative_accuracy": 0.68,
        "minimum_sequence_pearson": 0.55,
        "minimum_sequence_negative_accuracy": 0.10,
    }
    gates = validation_gates(
        baseline=baseline,
        routed=routed,
        dependence={
            "zero_event_pearson_drop": 0.60,
            "shuffled_event_pearson_drop": 0.62,
        },
        bootstrap_lower=0.58,
        selection_feasible=True,
        config=config,
    )
    assert gates["pearson_floor"] is True
    assert gates["negative_accuracy"] is True
    assert gates["minimum_sequence_negative_accuracy"] is False
