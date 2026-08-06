from __future__ import annotations

import numpy as np

from e_jepa_ttc.object_event_v4_4 import (
    fit_weighted_ridge,
    official_eap_metrics,
    sequence_sign_weights,
)


def test_weighted_ridge_recovers_linear_relation() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(256, 3))
    target = 0.02 + features @ np.asarray([0.4, -0.2, 0.1])
    model = fit_weighted_ridge(features, target, alpha=1.0e-8)
    prediction = model.predict(features)
    assert np.max(np.abs(prediction - target)) < 1.0e-6


def test_sequence_sign_weights_balance_cells() -> None:
    sequence = ["a"] * 6 + ["b"] * 4
    target = np.asarray([1, 1, 1, 1, -1, -1, 1, 1, 1, -1], dtype=np.float64)
    weights = sequence_sign_weights(sequence, target, cap=1000.0)
    cells = np.asarray(
        [f"{seq}|{'negative' if value < 0 else 'positive'}" for seq, value in zip(sequence, target, strict=True)]
    )
    masses = [float(weights[cells == cell].sum()) for cell in np.unique(cells)]
    assert np.max(masses) - np.min(masses) < 1.0e-10


def test_official_eap_metrics_are_zero_for_exact_prediction() -> None:
    target_ttc = np.asarray([2.0, 4.0, 8.0, -4.0], dtype=np.float64)
    delta_t = np.full(4, 0.1, dtype=np.float64)
    expansion = delta_t / target_ttc
    metrics = official_eap_metrics(expansion, expansion, delta_t, target_ttc)
    assert metrics["all_ranges_present"] is True
    assert np.isclose(float(metrics["weighted_mid"]), 0.0)
    assert np.isclose(float(metrics["weighted_rte_percent"]), 0.0)
