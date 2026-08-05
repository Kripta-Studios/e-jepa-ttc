from __future__ import annotations

import numpy as np

from scripts.train_e_jepa_object_event_v4_2 import (
    ObjectEventV42TrainConfig,
    _gates,
    _pearson,
)


def test_v42_pearson_returns_zero_for_constant_or_nonfinite_predictions() -> None:
    target = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    assert _pearson(target, np.zeros_like(target)) == 0.0
    assert _pearson(target, np.asarray([np.nan, 0.0, 1.0])) == 1.0


def test_v42_gates_treat_missing_dependence_as_fail_closed() -> None:
    metrics = {
        "event": {
            "pearson": 0.3,
            "balanced_sign_accuracy": 0.6,
            "negative_accuracy": 0.6,
            "expansion_mae": 0.01,
            "ttc_saturation_rate": 0.0,
        },
        "bootstrap_pearson": {"lower_95": 0.1},
        "per_sequence": {"macro_pearson": 0.2, "minimum_pearson": 0.1},
        "event_dependence": {
            "zero_event_pearson_drop": None,
            "shuffled_event_pearson_drop": None,
            "shuffled_event_mean_abs_change": None,
        },
    }
    gates = _gates(metrics, ObjectEventV42TrainConfig())
    assert gates["zero_event_dependence"] is False
    assert gates["shuffled_event_dependence"] is False
    assert gates["shuffled_event_change"] is False
