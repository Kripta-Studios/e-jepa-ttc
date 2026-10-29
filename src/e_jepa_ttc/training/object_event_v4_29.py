"""Losses, teacher invariants and fail-closed selection helpers for v4.29."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_29 import ObjectEventV429Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion
from e_jepa_ttc.training.object_event_v4_27 import balanced_sign_weights, target_log_height_ratio


@dataclass(frozen=True)
class ObjectEventV429LossConfig:
    arm: str = "local_affine_lhr"
    lhr_weight: float = 4.0
    expansion_weight: float = 1.0
    correlation_weight: float = 1.0
    sign_weight: float = 1.0
    confidence_weight: float = 0.05
    composition_weight: float = 0.20
    residual_weight: float = 0.05
    invalid_weight: float = 2.0
    geometry_teacher_weight: float = 0.50
    smooth_l1_beta: float = 0.004
    sign_temperature: float = 0.015
    max_abs_expansion: float = 0.25
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.arm not in {"local_affine_lhr", "local_affine_geom_teacher"}:
            raise ValueError("v4.29 allows exactly the two preregistered arms")
        if (
            min(
                self.lhr_weight,
                self.expansion_weight,
                self.correlation_weight,
                self.sign_weight,
                self.confidence_weight,
                self.composition_weight,
                self.residual_weight,
                self.invalid_weight,
                self.geometry_teacher_weight,
            )
            < 0.0
        ):
            raise ValueError("v4.29 loss weights must be non-negative")
        if min(self.smooth_l1_beta, self.sign_temperature, self.epsilon) <= 0.0:
            raise ValueError("v4.29 numerical loss parameters must be positive")


def common_roi_box_invariants(
    boxes_xyxy: torch.Tensor, *, height: int, width: int, epsilon: float = 1e-6
) -> dict[str, torch.Tensor]:
    """Teacher targets using only t1/t2 boxes in common-ROI coordinates.

    ``boxes_xyxy`` is ``[B,3,4]`` in input pixels.  t0 is intentionally not
    used: its box is a proxy and is not a valid teacher target for v4.29.
    """
    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[1:] != (3, 4):
        raise ValueError("boxes_xyxy must be [B,3,4]")
    if height <= 1 or width <= 1:
        raise ValueError("teacher ROI dimensions must exceed one pixel")
    b1, b2 = boxes_xyxy[:, 1], boxes_xyxy[:, 2]
    h1 = (b1[:, 3] - b1[:, 1]).clamp_min(epsilon)
    h2 = (b2[:, 3] - b2[:, 1]).clamp_min(epsilon)
    w1 = (b1[:, 2] - b1[:, 0]).clamp_min(epsilon)
    w2 = (b2[:, 2] - b2[:, 0]).clamp_min(epsilon)

    def center(box: torch.Tensor) -> torch.Tensor:
        # Cache boxes use ROI edge coordinates, matching the feature-grid convention.
        x = ((box[:, 0] + box[:, 2]) * 0.5) * (2.0 / float(width)) - 1.0
        y = ((box[:, 1] + box[:, 3]) * 0.5) * (2.0 / float(height)) - 1.0
        return torch.stack((x, y), dim=-1)

    return {
        "box_log_height_ratio_t1_t2": h1.log() - h2.log(),
        "box_log_width_ratio_t1_t2": w1.log() - w2.log(),
        "center_t1": center(b1),
        "center_t2": center(b2),
    }


def _masked_mean(value: torch.Tensor, valid: torch.Tensor, epsilon: float) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(epsilon)


def object_event_v4_29_loss(
    output: ObjectEventV429Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    *,
    config: ObjectEventV429LossConfig,
    boxes_xyxy: torch.Tensor | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train with labels only outside the model forward path.

    Invalid affine predictions are excluded from supervised terms. Differentiable
    determinant/condition/support barriers discourage invalid fits, while the
    discrete invalid fraction remains diagnostic. Invalid predictions are never
    converted to plausible finite values.
    """
    target_exp = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    target_lhr = target_log_height_ratio(visible_heights_px)
    valid = output.affine_12.valid & torch.isfinite(output.predicted_log_eta_vertical)
    safe_log = torch.where(valid, output.predicted_log_eta_vertical, torch.zeros_like(target_lhr))
    safe_exp = torch.where(valid, output.expansion, torch.zeros_like(target_exp))
    sample = balanced_sign_weights(target_exp, config.epsilon)
    lhr_each = functional.smooth_l1_loss(
        safe_log, target_lhr, beta=config.smooth_l1_beta, reduction="none"
    )
    exp_each = functional.smooth_l1_loss(
        safe_exp, target_exp, beta=config.smooth_l1_beta, reduction="none"
    )
    signed = torch.where(
        target_exp >= 0.0, torch.ones_like(target_exp), -torch.ones_like(target_exp)
    )
    sign_each = functional.softplus(-signed * safe_exp / config.sign_temperature)
    weighted_valid = valid.to(sample.dtype) * sample
    lhr = (weighted_valid * lhr_each).sum() / weighted_valid.sum().clamp_min(config.epsilon)
    expansion = (weighted_valid * exp_each).sum() / weighted_valid.sum().clamp_min(config.epsilon)
    sign = (weighted_valid * sign_each).sum() / weighted_valid.sum().clamp_min(config.epsilon)
    correlation = (
        1.0 - pearson_torch(safe_exp[valid], target_exp[valid])
        if int(valid.sum()) >= 2
        else safe_exp.sum() * 0.0 + 1.0
    )
    confidence = (1.0 - output.correlation_confidence).mean() + output.boundary_probability.mean()
    composition = (
        output.composition_matrix_error.mean() + output.composition_translation_error.mean()
    )
    residual = output.affine_12.residual.mean()
    invalid_fraction = 1.0 - valid.to(target_exp.dtype).mean()
    invalid = output.validity_penalty.mean()
    teacher = safe_exp.sum() * 0.0
    if config.arm == "local_affine_geom_teacher":
        if boxes_xyxy is None or image_height is None or image_width is None:
            raise ValueError(
                "geometry teacher arm requires train-only t1/t2 boxes and image dimensions"
            )
        targets = common_roi_box_invariants(
            boxes_xyxy, height=image_height, width=image_width, epsilon=config.epsilon
        )
        predicted_center = (output.affine_12.matrix @ targets["center_t2"][..., None]).squeeze(
            -1
        ) + output.affine_12.translation
        center_error = torch.linalg.vector_norm(predicted_center - targets["center_t1"], dim=-1)
        teacher = _masked_mean(
            functional.smooth_l1_loss(
                safe_log, targets["box_log_height_ratio_t1_t2"], beta=0.02, reduction="none"
            ),
            valid,
            config.epsilon,
        )
        safe_horizontal = torch.where(
            valid, output.predicted_log_eta_horizontal, torch.zeros_like(safe_log)
        )
        teacher = teacher + _masked_mean(
            functional.smooth_l1_loss(
                safe_horizontal, targets["box_log_width_ratio_t1_t2"], beta=0.02, reduction="none"
            ),
            valid,
            config.epsilon,
        )
        teacher = teacher + _masked_mean(center_error, valid, config.epsilon)
    pieces = {
        "lhr": lhr,
        "expansion": expansion,
        "correlation": correlation,
        "sign": sign,
        "confidence": confidence,
        "composition": composition,
        "residual": residual,
        "invalid": invalid,
        "invalid_fraction": invalid_fraction,
        "geometry_teacher": teacher,
        "target_log_eta": target_lhr,
        "target_expansion": target_exp,
        "valid_fraction": valid.to(target_exp.dtype).mean(),
    }
    total = (
        config.lhr_weight * lhr
        + config.expansion_weight * expansion
        + config.correlation_weight * correlation
        + config.sign_weight * sign
        + config.confidence_weight * confidence
        + config.composition_weight * composition
        + config.residual_weight * residual
        + config.invalid_weight * invalid
        + (
            config.geometry_teacher_weight * teacher
            if config.arm == "local_affine_geom_teacher"
            else 0.0
        )
    )
    return total, pieces


def seed_dominance(
    matcher_level_means: Mapping[int, float],
    backbone_level_means: Mapping[int, float],
    *,
    material: float = 0.03,
) -> str:
    """Preregistered descriptive 3x3 factorial dominance rule, not variance inference."""
    mr = max(matcher_level_means.values()) - min(matcher_level_means.values())
    br = max(backbone_level_means.values()) - min(backbone_level_means.values())
    if mr < material and br < material:
        return "neither_inconclusive"
    if mr >= material and br >= material:
        if max(mr, br) / min(mr, br) < 1.5:
            return "both_material"
        return "matcher_dominant" if mr > br else "backbone_dominant"
    return "matcher_dominant" if mr >= material else "backbone_dominant"


def oof_gates(
    metrics: Mapping[str, float],
    reference_v428: Mapping[str, float],
    tracks: Mapping[str, float],
    reference_v427: Mapping[str, float] | None = None,
    reference_v428_tracks: Mapping[str, float] | None = None,
) -> dict[str, bool]:
    """All v4.29 OOF gates. Missing/non-finite values fail closed."""
    required = (
        "pearson",
        "negative_accuracy",
        "balanced_sign_accuracy",
        "log_eta_pearson",
        "minimum_sequence_pearson",
        "prediction_std_ratio",
        "calibration_slope_intercept",
        "invalid_affine_fraction",
        "high_magnitude_ratio",
        "high_magnitude_count",
    )
    if any(
        k not in metrics or not torch.isfinite(torch.tensor(float(metrics[k]))) for k in required
    ):
        return {"complete_finite": False}
    neg_track = float(tracks.get("negative_track_macro_accuracy", float("nan")))
    checks = {
        "complete_finite": bool(metrics.get("complete_finite", False))
        and torch.isfinite(torch.tensor(neg_track)).item(),
        "pearson": float(metrics["pearson"]) >= 0.635,
        "negative_accuracy": float(metrics["negative_accuracy"]) >= 0.652021,
        "balanced_sign": float(metrics["balanced_sign_accuracy"]) >= 0.775,
        "log_eta": float(metrics["log_eta_pearson"]) >= 0.615,
        "negative_track_macro": neg_track >= 0.697674,
        "minimum_sequence": float(metrics["minimum_sequence_pearson"]) >= 0.430,
        "std_ratio": 0.75 <= float(metrics["prediction_std_ratio"]) <= 1.25,
        "calibration": 0.70 <= float(metrics["calibration_slope_intercept"]) <= 1.30,
        "invalid": float(metrics["invalid_affine_fraction"]) <= 0.02,
        "high_magnitude": float(metrics["high_magnitude_count"]) > 0
        and 0.70 <= float(metrics["high_magnitude_ratio"]) <= 1.30,
        "pearson_gain_v428": float(metrics["pearson"]) >= float(reference_v428["pearson"]) + 0.025,
        "log_eta_gain_v428": float(metrics["log_eta_pearson"])
        >= float(reference_v428["log_eta_pearson"]) + 0.015,
        "negative_gain_v428": float(metrics["negative_accuracy"])
        >= float(reference_v428["negative_accuracy"]) + 0.020,
        "min_sequence_gain_v428": float(metrics["minimum_sequence_pearson"])
        >= float(reference_v428["minimum_sequence_pearson"]) + 0.100,
    }
    if reference_v427 is not None:
        checks["negative_at_least_v427"] = float(metrics["negative_accuracy"]) >= float(
            reference_v427["negative_accuracy"]
        )
    if reference_v428_tracks is not None:
        checks["negative_track_at_least_v428"] = neg_track >= float(
            reference_v428_tracks["negative_track_macro_accuracy"]
        )
    return {key: bool(value) for key, value in checks.items()}


__all__ = [
    "ObjectEventV429LossConfig",
    "common_roi_box_invariants",
    "object_event_v4_29_loss",
    "oof_gates",
    "seed_dominance",
]
