from __future__ import annotations

from scripts.train_e_jepa_object_event_v4_7 import (
    ObjectEventV47TrainConfig,
    _epoch_gates,
    _selection_objective,
)


def _metrics(*, mask_iou: float, minimum_sequence_pearson: float = -1.0) -> dict:
    return {
        "event": {
            "pearson": 0.985,
            "balanced_sign_accuracy": 1.0,
            "negative_accuracy": 1.0,
        },
        "geometry": {
            "height_log_eta_pearson": 0.975,
            "height_log_eta_mae": 0.007,
            "foreground_soft_iou": mask_iou,
            "minimum_sequence_height_pearson": minimum_sequence_pearson,
        },
        "per_sequence": {"minimum_eligible_negative_accuracy": 0.0},
        "event_dependence": {
            "zero_event_pearson_drop": 0.9,
            "shuffled_event_pearson_drop": 0.9,
        },
    }


def test_overfit_selection_ignores_sparse_per_sequence_terms() -> None:
    a = _metrics(mask_iou=0.81, minimum_sequence_pearson=-1.0)
    b = _metrics(mask_iou=0.81, minimum_sequence_pearson=0.9)
    assert _selection_objective(a, mode="overfit") == _selection_objective(
        b, mode="overfit"
    )
    assert _selection_objective(a, mode="screen") > _selection_objective(
        b, mode="screen"
    )


def test_epoch_gates_accept_gate_passing_overfit_checkpoint() -> None:
    config = ObjectEventV47TrainConfig()
    gates = _epoch_gates(_metrics(mask_iou=0.81), config, mode="overfit")
    assert gates == {
        "expansion_pearson": True,
        "balanced_sign": True,
        "height_ratio": True,
        "foreground": True,
    }
