"""Synthetic mechanistic gates for the v5 causal foreground-scale operator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional

from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    soft_vertical_extent_from_logits,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "analytic_pearson_min": 0.95,
    "slope_min": 0.8,
    "slope_max": 1.2,
    "sign_accuracy_min": 0.95,
    "oddness_median_max": 0.2,
    "oddness_p95_max": 0.5,
    "identity_p95_max": 1.0e-5,
    "translation_leakage_p95_max": 1.0e-4,
    "rotation_leakage_p95_max": 0.02,
    "zero_unknown_min": 1.0,
}

SYNTHETIC_LEARNING_THRESHOLDS: dict[str, float] = {
    "analytic_pearson_min": 0.95,
    "slope_min": 0.8,
    "slope_max": 1.2,
    "sign_accuracy_min": 0.95,
    "foreground_iou_min": 0.60,
    "known_coverage_min": 0.90,
    "empty_unknown_min": 1.0,
    "empty_false_positive_fraction_max": 0.01,
    "oddness_median_max": 0.05,
    "oddness_p95_max": 0.10,
    "translation_leakage_p95_max": 0.02,
    "ttc_symmetric_relative_error_max": 0.30,
    "ratio_80_coverage_min": 0.60,
    "ratio_80_coverage_max": 0.95,
}


def _rectangle_logits(
    height: int,
    width: int,
    *,
    canvas: int = 128,
    center_y: int = 64,
    center_x: int = 64,
) -> torch.Tensor:
    if min(height, width) <= 1 or max(height, width) >= canvas:
        raise ValueError("rectangle must fit strictly inside the canvas")
    logits = torch.full((1, 1, canvas, canvas), -20.0)
    top = center_y - height // 2
    left = center_x - width // 2
    if top < 0 or left < 0 or top + height > canvas or left + width > canvas:
        raise ValueError("rectangle placement escapes the canvas")
    logits[..., top : top + height, left : left + width] = 20.0
    return logits


def _log_extent(logits: torch.Tensor) -> torch.Tensor:
    return soft_vertical_extent_from_logits(logits).height_normalized.log()


def _percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q / 100.0).item())


def _rotation(logits: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    theta = logits.new_tensor(
        [
            [
                [math.cos(radians), -math.sin(radians), 0.0],
                [math.sin(radians), math.cos(radians), 0.0],
            ]
        ]
    )
    grid = functional.affine_grid(theta, list(logits.shape), align_corners=False)
    return functional.grid_sample(
        logits,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def synthetic_operator_metrics(
    model_config: CausalScaleTTCConfig | None = None,
    *,
    seed: int = 7,
) -> dict[str, float]:
    """Measure scale algebra and fail-safe behavior without real data or labels."""

    reciprocal_pairs = (
        ((40, 32), (32, 40)),
        ((36, 32), (32, 36)),
        ((48, 32), (32, 48)),
        ((44, 32), (32, 44)),
    )
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    oddness: list[torch.Tensor] = []
    for forward_pair, reverse_pair in reciprocal_pairs:
        pair_predictions: list[torch.Tensor] = []
        for previous_height, current_height in (forward_pair, reverse_pair):
            previous = _log_extent(_rectangle_logits(previous_height, 40))
            current = _log_extent(_rectangle_logits(current_height, 40))
            target = previous.new_tensor(math.log(current_height / previous_height))
            prediction = current - previous
            targets.append(target)
            predictions.append(prediction)
            pair_predictions.append(prediction)
        numerator = (pair_predictions[0] + pair_predictions[1]).abs()
        denominator = pair_predictions[0].abs() + pair_predictions[1].abs()
        oddness.append(numerator / denominator.clamp_min(1.0e-12))
    target_tensor = torch.stack(targets).flatten()
    prediction_tensor = torch.stack(predictions).flatten()
    correlation = torch.corrcoef(torch.stack((target_tensor, prediction_tensor)))[0, 1]
    slope = torch.dot(target_tensor, prediction_tensor) / torch.dot(target_tensor, target_tensor)
    sign_accuracy = (torch.sign(target_tensor) == torch.sign(prediction_tensor)).float().mean()
    oddness_tensor = torch.stack(oddness).flatten()

    identity_errors: list[torch.Tensor] = []
    translation_errors: list[torch.Tensor] = []
    reference = _log_extent(_rectangle_logits(32, 40, center_y=64))
    for center in (44, 54, 64, 74, 84):
        current = _log_extent(_rectangle_logits(32, 40, center_y=center))
        identity_errors.append((current - current).abs())
        translation_errors.append((current - reference).abs())
    square = _rectangle_logits(32, 32)
    square_extent = _log_extent(square)
    rotation_errors = torch.stack(
        [(_log_extent(_rotation(square, angle)) - square_extent).abs() for angle in (-5.0, 5.0)]
    ).flatten()

    torch.manual_seed(seed)
    config = model_config or CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=16,
        geometry_dim=24,
        residual_depth=1,
        dropout=0.0,
    )
    model = CausalScaleTTC(config).eval()
    zeros = torch.zeros(4, 3, config.in_channels, 32, 32)
    with torch.inference_mode():
        zero_output = model(zeros, torch.full((4, 2), 0.1))
    return {
        "analytic_pearson": float(correlation.item()),
        "slope": float(slope.item()),
        "sign_accuracy": float(sign_accuracy.item()),
        "oddness_median": float(oddness_tensor.median().item()),
        "oddness_p95": _percentile(oddness_tensor, 95.0),
        "identity_p95": _percentile(torch.stack(identity_errors).flatten(), 95.0),
        "translation_leakage_p95": _percentile(
            torch.stack(translation_errors).flatten(),
            95.0,
        ),
        "rotation_leakage_p95": _percentile(rotation_errors, 95.0),
        "zero_unknown": float((~zero_output.known_mask).float().mean().item()),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
    }


def evaluate_operator_gates(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict[str, bool]:
    """Apply the preregistered v5 synthetic operator thresholds fail-closed."""

    required_metrics = {
        "analytic_pearson",
        "slope",
        "sign_accuracy",
        "oddness_median",
        "oddness_p95",
        "identity_p95",
        "translation_leakage_p95",
        "rotation_leakage_p95",
        "zero_unknown",
    }
    finite = all(key in metrics and math.isfinite(float(metrics[key])) for key in required_metrics)
    if not finite:
        return {"finite": False, "passed": False}
    gates = {
        "finite": True,
        "equivariance": metrics["analytic_pearson"] >= thresholds["analytic_pearson_min"]
        and thresholds["slope_min"] <= metrics["slope"] <= thresholds["slope_max"]
        and metrics["sign_accuracy"] >= thresholds["sign_accuracy_min"],
        "oddness": metrics["oddness_median"] <= thresholds["oddness_median_max"]
        and metrics["oddness_p95"] <= thresholds["oddness_p95_max"],
        "identity": metrics["identity_p95"] <= thresholds["identity_p95_max"],
        "translation": metrics["translation_leakage_p95"]
        <= thresholds["translation_leakage_p95_max"],
        "rotation": metrics["rotation_leakage_p95"] <= thresholds["rotation_leakage_p95_max"],
        "zero": metrics["zero_unknown"] >= thresholds["zero_unknown_min"],
    }
    gates["passed"] = all(value for key, value in gates.items() if key != "finite")
    return gates


def validate_thresholds(values: Mapping[str, Any]) -> dict[str, float]:
    """Require the exact named gate set before running the synthetic protocol."""

    if set(values) != set(DEFAULT_THRESHOLDS):
        missing = sorted(set(DEFAULT_THRESHOLDS) - set(values))
        extra = sorted(set(values) - set(DEFAULT_THRESHOLDS))
        raise ValueError(f"operator threshold keys differ; missing={missing}, extra={extra}")
    thresholds = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise ValueError("operator thresholds must be finite")
    return thresholds


def evaluate_synthetic_learning_gates(
    metrics: Mapping[str, float | None],
    thresholds: Mapping[str, float] = SYNTHETIC_LEARNING_THRESHOLDS,
) -> dict[str, bool]:
    """Fail closed on learned foreground, scale, safety, controls, and calibration."""

    required = {
        "analytic_pearson",
        "slope",
        "sign_accuracy",
        "foreground_iou",
        "known_coverage",
        "empty_unknown",
        "empty_false_positive_fraction",
        "oddness_median",
        "oddness_p95",
        "translation_leakage_p95",
        "ttc_symmetric_relative_error",
        "ratio_80_coverage",
    }
    finite = all(
        isinstance(metrics.get(key), (int, float))
        and math.isfinite(float(metrics[key]))  # type: ignore[arg-type]
        for key in required
    )
    if not finite:
        return {"finite": False, "passed": False}
    values = {key: float(metrics[key]) for key in required}  # type: ignore[arg-type]
    gates = {
        "finite": True,
        "equivariance": values["analytic_pearson"] >= thresholds["analytic_pearson_min"]
        and thresholds["slope_min"] <= values["slope"] <= thresholds["slope_max"]
        and values["sign_accuracy"] >= thresholds["sign_accuracy_min"],
        "foreground": values["foreground_iou"] >= thresholds["foreground_iou_min"],
        "known": values["known_coverage"] >= thresholds["known_coverage_min"],
        "empty": values["empty_unknown"] >= thresholds["empty_unknown_min"]
        and values["empty_false_positive_fraction"]
        <= thresholds["empty_false_positive_fraction_max"],
        "oddness": values["oddness_median"] <= thresholds["oddness_median_max"]
        and values["oddness_p95"] <= thresholds["oddness_p95_max"],
        "translation": values["translation_leakage_p95"]
        <= thresholds["translation_leakage_p95_max"],
        "ttc": values["ttc_symmetric_relative_error"]
        <= thresholds["ttc_symmetric_relative_error_max"],
        "calibration": thresholds["ratio_80_coverage_min"]
        <= values["ratio_80_coverage"]
        <= thresholds["ratio_80_coverage_max"],
    }
    gates["passed"] = all(value for key, value in gates.items() if key != "finite")
    return gates


__all__ = [
    "DEFAULT_THRESHOLDS",
    "SYNTHETIC_LEARNING_THRESHOLDS",
    "evaluate_operator_gates",
    "evaluate_synthetic_learning_gates",
    "synthetic_operator_metrics",
    "validate_thresholds",
]
