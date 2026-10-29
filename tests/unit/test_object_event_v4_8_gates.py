from __future__ import annotations

from scripts.train_e_jepa_object_event_v4_8 import ObjectEventV48TrainConfig, _gates


def test_v48_overfit_gates_require_dense_motion_learning() -> None:
    metrics = {
        "event": {"pearson": 0.98, "balanced_sign_accuracy": 0.97},
        "motion_field": {
            "log_eta_pearson": 0.98,
            "log_eta_mae": 0.004,
            "dense_foreground_mae": 0.020,
            "foreground_soft_iou": 0.72,
        },
        "event_dependence": {},
    }
    gates = _gates(metrics, ObjectEventV48TrainConfig(), mode="overfit")
    assert gates["pooled_log_eta_mae"]
    assert "dense_mae" not in gates
    assert all(gates.values())


def test_v48_overfit_gates_fail_on_bad_supervised_scalar_mae() -> None:
    metrics = {
        "event": {"pearson": 0.98, "balanced_sign_accuracy": 0.97},
        "motion_field": {
            "log_eta_pearson": 0.98,
            "log_eta_mae": 0.02,
            "dense_foreground_mae": 0.001,
            "foreground_soft_iou": 0.72,
        },
        "event_dependence": {},
    }
    gates = _gates(metrics, ObjectEventV48TrainConfig(), mode="overfit")
    assert not gates["pooled_log_eta_mae"]
