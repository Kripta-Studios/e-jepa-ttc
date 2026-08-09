from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from e_jepa_ttc.data.synthetic_causal_scale import (
    SyntheticCausalScaleConfig,
    SyntheticCausalScaleDataset,
)
from e_jepa_ttc.evaluation.causal_scale_v5 import (
    SYNTHETIC_LEARNING_THRESHOLDS,
    evaluate_synthetic_learning_gates,
)
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_v5 import (
    CausalScaleSyntheticTrainingConfig,
    calibrate_ratio_uncertainty,
    checkpoint_payload,
    evaluate_synthetic_causal_scale,
    train_synthetic_causal_scale,
)


def _dataset(seed: int, samples: int) -> SyntheticCausalScaleDataset:
    return SyntheticCausalScaleDataset(
        SyntheticCausalScaleConfig(
            samples=samples,
            seed=seed,
            canvas_size=32,
            background_events_per_endpoint=1,
            hot_pixel_probability=0.0,
            empty_probability=0.125,
        )
    )


def test_synthetic_training_returns_finite_validation_selected_checkpoint() -> None:
    train = _dataset(11, 8)
    validation = _dataset(22, 8)
    model_config = CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=8,
        geometry_dim=16,
        residual_depth=1,
        dropout=0.0,
    )
    training_config = CausalScaleSyntheticTrainingConfig(
        seed=5,
        epochs=1,
        batch_size=4,
    )
    loss_config = CausalScaleTTCLossConfig(temporal_consistency_weight=0.0)

    result = train_synthetic_causal_scale(
        model_config,
        training_config,
        loss_config,
        train,
        validation,
        torch.device("cpu"),
    )
    calibration = calibrate_ratio_uncertainty(
        result.model,
        DataLoader(validation, batch_size=4),
        torch.device("cpu"),
    )
    metrics = evaluate_synthetic_causal_scale(
        result.model,
        DataLoader(validation, batch_size=4),
        torch.device("cpu"),
        loss_config=loss_config,
        controls=True,
    )
    payload = checkpoint_payload(result, training_config, loss_config)

    assert result.best_epoch == 1
    assert math.isfinite(result.best_selection_score)
    assert metrics["analytic_pearson"] is not None
    for row in result.history:
        train_total = row["train_total"]
        assert isinstance(train_total, (int, float))
        assert math.isfinite(float(train_total))
    assert payload["artifact_type"] == "causal_scale_v5_synthetic_checkpoint_v1"
    assert payload["best_epoch"] == 1
    assert calibration["valid_count"] > 0
    assert math.isfinite(float(calibration["log_variance_offset"]))


def test_synthetic_learning_gates_fail_closed_and_accept_exact_thresholds() -> None:
    assert evaluate_synthetic_learning_gates({}, SYNTHETIC_LEARNING_THRESHOLDS) == {
        "finite": False,
        "passed": False,
    }
    passing = {
        "analytic_pearson": SYNTHETIC_LEARNING_THRESHOLDS["analytic_pearson_min"],
        "slope": SYNTHETIC_LEARNING_THRESHOLDS["slope_min"],
        "sign_accuracy": SYNTHETIC_LEARNING_THRESHOLDS["sign_accuracy_min"],
        "foreground_iou": SYNTHETIC_LEARNING_THRESHOLDS["foreground_iou_min"],
        "known_coverage": SYNTHETIC_LEARNING_THRESHOLDS["known_coverage_min"],
        "empty_unknown": SYNTHETIC_LEARNING_THRESHOLDS["empty_unknown_min"],
        "empty_false_positive_fraction": SYNTHETIC_LEARNING_THRESHOLDS[
            "empty_false_positive_fraction_max"
        ],
        "oddness_median": SYNTHETIC_LEARNING_THRESHOLDS["oddness_median_max"],
        "oddness_p95": SYNTHETIC_LEARNING_THRESHOLDS["oddness_p95_max"],
        "translation_leakage_p95": SYNTHETIC_LEARNING_THRESHOLDS[
            "translation_leakage_p95_max"
        ],
        "ttc_symmetric_relative_error": SYNTHETIC_LEARNING_THRESHOLDS[
            "ttc_symmetric_relative_error_max"
        ],
        "ratio_80_coverage": SYNTHETIC_LEARNING_THRESHOLDS["ratio_80_coverage_min"],
    }

    result = evaluate_synthetic_learning_gates(passing, SYNTHETIC_LEARNING_THRESHOLDS)

    assert result["finite"]
    assert result["passed"]
