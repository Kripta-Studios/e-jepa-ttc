from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.aggregate_object_event_v4_3_multiseed import (
    _branch_metrics,
    _pearson,
    _track_cluster_bootstrap,
)


def _frame() -> pd.DataFrame:
    target = np.asarray([-0.04, -0.02, 0.01, 0.03, -0.03, 0.02], dtype=np.float64)
    prediction = np.asarray([-0.03, -0.01, 0.015, 0.025, -0.02, 0.018], dtype=np.float64)
    return pd.DataFrame({
        "sequence_id": ["a", "a", "a", "b", "b", "b"],
        "track_id": ["a1", "a1", "a2", "b1", "b1", "b2"],
        "delta_t_s": np.full(6, 0.1),
        "target_expansion": target,
        "ensemble_expansion": prediction,
    })


def test_v43_metrics_are_finite_and_sign_aware() -> None:
    frame = _frame()
    metrics = _branch_metrics(frame, "ensemble_expansion")
    assert metrics["pearson"] > 0.95
    assert metrics["negative_accuracy"] == 1.0
    assert metrics["balanced_sign_accuracy"] == 1.0
    assert _pearson(np.ones(3), np.ones(3)) == 0.0


def test_v43_track_cluster_bootstrap_is_reproducible() -> None:
    frame = _frame()
    first = _track_cluster_bootstrap(
        frame, prediction_column="ensemble_expansion", repeats=200, seed=7
    )
    second = _track_cluster_bootstrap(
        frame, prediction_column="ensemble_expansion", repeats=200, seed=7
    )
    assert first == second
    assert first["cluster_count"] == 4
    assert 0.0 <= first["lower_95"] <= 1.0
