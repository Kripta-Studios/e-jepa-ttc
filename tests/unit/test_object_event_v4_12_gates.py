import pytest

from e_jepa_ttc.training.object_event_v4_12 import (
    directional_sign_checkpoint_gates,
    directional_sign_gates,
)


def test_screen_gates_reject_sequence_level_negative_failure() -> None:
    baseline = {"pearson": 0.676, "expansion_mae": 0.0149}
    metrics = {
        "pearson": 0.67,
        "expansion_mae": 0.0150,
        "balanced_sign_accuracy": 0.78,
        "negative_accuracy": 0.69,
        "minimum_sequence_negative_accuracy": 0.20,
        "reverse_sign_accuracy": 0.90,
        "zero_event_pearson_drop": 0.67,
        "shuffled_event_pearson_drop": 0.68,
    }
    thresholds = {
        "screen_pearson_floor": 0.63,
        "screen_pearson_max_drop": 0.025,
        "screen_mae_tolerance": 0.0015,
        "screen_balanced_sign_gate": 0.76,
        "screen_negative_accuracy_gate": 0.66,
        "screen_min_sequence_negative_accuracy_gate": 0.30,
        "screen_reverse_accuracy_gate": 0.80,
        "zero_event_pearson_drop_gate": 0.55,
        "shuffled_event_pearson_drop_gate": 0.55,
    }
    gates = directional_sign_gates(
        mode="screen", metrics=metrics, baseline=baseline, thresholds=thresholds
    )
    assert not gates["minimum_sequence_negative_accuracy"]
    assert not all(gates.values())


def test_overfit_gates_require_reversal_antisymmetry() -> None:
    metrics = {
        "balanced_sign_accuracy": 0.99,
        "negative_accuracy": 0.99,
        "reverse_sign_accuracy": 0.99,
        "antisymmetry_mean_abs": 0.5,
    }
    thresholds = {
        "overfit_balanced_sign_gate": 0.95,
        "overfit_negative_accuracy_gate": 0.95,
        "overfit_reverse_accuracy_gate": 0.95,
        "overfit_antisymmetry_ceiling": 0.35,
    }
    gates = directional_sign_gates(
        mode="overfit", metrics=metrics, baseline={}, thresholds=thresholds
    )
    assert not gates["antisymmetry"]


def test_screen_checkpoint_gates_do_not_require_expensive_dependence_metrics() -> None:
    baseline = {"pearson": 0.676, "expansion_mae": 0.0149}
    metrics = {
        "pearson": 0.67,
        "expansion_mae": 0.0150,
        "balanced_sign_accuracy": 0.78,
        "negative_accuracy": 0.69,
        "minimum_sequence_negative_accuracy": 0.35,
        "reverse_sign_accuracy": 0.90,
    }
    thresholds = {
        "screen_pearson_floor": 0.63,
        "screen_pearson_max_drop": 0.025,
        "screen_mae_tolerance": 0.0015,
        "screen_balanced_sign_gate": 0.76,
        "screen_negative_accuracy_gate": 0.66,
        "screen_min_sequence_negative_accuracy_gate": 0.30,
        "screen_reverse_accuracy_gate": 0.80,
    }
    gates = directional_sign_checkpoint_gates(
        mode="screen", metrics=metrics, baseline=baseline, thresholds=thresholds
    )
    assert all(gates.values())
    assert "zero_event_dependence" not in gates
    assert "shuffled_event_dependence" not in gates


def test_final_screen_gates_still_require_event_dependence_metrics() -> None:
    baseline = {"pearson": 0.676, "expansion_mae": 0.0149}
    metrics = {
        "pearson": 0.67,
        "expansion_mae": 0.0150,
        "balanced_sign_accuracy": 0.78,
        "negative_accuracy": 0.69,
        "minimum_sequence_negative_accuracy": 0.35,
        "reverse_sign_accuracy": 0.90,
    }
    thresholds = {
        "screen_pearson_floor": 0.63,
        "screen_pearson_max_drop": 0.025,
        "screen_mae_tolerance": 0.0015,
        "screen_balanced_sign_gate": 0.76,
        "screen_negative_accuracy_gate": 0.66,
        "screen_min_sequence_negative_accuracy_gate": 0.30,
        "screen_reverse_accuracy_gate": 0.80,
        "zero_event_pearson_drop_gate": 0.55,
        "shuffled_event_pearson_drop_gate": 0.55,
    }
    with pytest.raises(KeyError, match="zero_event_pearson_drop"):
        directional_sign_gates(
            mode="screen", metrics=metrics, baseline=baseline, thresholds=thresholds
        )
