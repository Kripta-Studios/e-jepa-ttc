from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.object_event_v4_9 import (
    FixedFusionConfig,
    add_fusion_columns,
    align_prediction_frames,
    convex_fusion,
    dependence_metrics,
    evaluate_prediction,
    fusion_gates,
)


def _frame(prediction: list[float]) -> pd.DataFrame:
    target = np.asarray([-0.04, -0.02, 0.02, 0.05], dtype=np.float64)
    return pd.DataFrame(
        {
            "sequence_id": ["a", "a", "b", "b"],
            "sample_token": ["0", "1", "2", "3"],
            "track_id": ["x", "x", "y", "y"],
            "delta_t_s": [0.05] * 4,
            "target_ttc_s": 0.05 / target,
            "target_expansion": target,
            "prediction_expansion": prediction,
            "zero_events_expansion": [0.0] * 4,
            "shuffled_mean_expansion": list(reversed(prediction)),
        }
    )


def test_convex_fusion_is_exact_and_bounded() -> None:
    base = np.asarray([-1.0, 1.0])
    dense = np.asarray([1.0, 3.0])
    assert np.allclose(convex_fusion(base, dense, 0.5), [0.0, 2.0])
    with pytest.raises(ValueError):
        convex_fusion(base, dense, 1.1)


def test_alignment_uses_identity_not_row_order() -> None:
    base = _frame([-0.03, -0.01, 0.01, 0.04])
    dense = _frame([-0.05, -0.02, 0.03, 0.05]).iloc[::-1]
    aligned = align_prediction_frames(base, dense, split_name="test")
    assert aligned["sample_token"].tolist() == ["0", "1", "2", "3"]
    assert np.allclose(aligned["dense_prediction_expansion"], [-0.05, -0.02, 0.03, 0.05])


def test_alignment_rejects_target_mismatch() -> None:
    base = _frame([-0.03, -0.01, 0.01, 0.04])
    dense = _frame([-0.05, -0.02, 0.03, 0.05])
    dense.loc[0, "target_expansion"] = 0.1
    with pytest.raises(ValueError, match="target mismatch"):
        align_prediction_frames(base, dense, split_name="test")


def test_fixed_fusion_metrics_and_dependence() -> None:
    config = FixedFusionConfig(
        per_sequence_negative_min_count=1,
        pearson_improvement_gate=0.0,
        mae_relative_improvement_gate=0.0,
        weighted_mid_relative_improvement_gate=0.0,
        balanced_sign_gate=0.5,
        negative_accuracy_gate=0.5,
        minimum_sequence_pearson_gate=0.0,
        minimum_sequence_negative_accuracy_gate=0.0,
        zero_event_pearson_drop_gate=0.1,
        shuffled_event_pearson_drop_gate=0.1,
    )
    aligned = align_prediction_frames(
        _frame([-0.01, 0.01, 0.01, 0.03]),
        _frame([-0.05, -0.03, 0.03, 0.06]),
        split_name="test",
    )
    rows = add_fusion_columns(aligned, alpha=0.5)
    base, _ = evaluate_prediction(rows, "base_prediction_expansion", config=config)
    dense, _ = evaluate_prediction(rows, "dense_prediction_expansion", config=config)
    fused, _ = evaluate_prediction(rows, "fused_prediction_expansion", config=config)
    dependence = dependence_metrics(rows)
    base["official_eap"]["weighted_mid"] = 100.0
    dense["official_eap"]["weighted_mid"] = 90.0
    fused["official_eap"]["weighted_mid"] = 80.0
    gates = fusion_gates(base=base, dense=dense, fused=fused, dependence=dependence, config=config)
    assert fused["balanced_sign_accuracy"] == 1.0
    assert dependence["zero_event_pearson_drop"] > 0.9
    assert all(gates.values())
