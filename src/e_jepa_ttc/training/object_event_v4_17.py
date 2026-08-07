"""Losses, anchor transforms and gates for Object Event TTC v4.17."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ObjectEventV417LossConfig:
    final_sign_weight: float = 1.0
    history_sign_weight: float = 0.35

    def __post_init__(self) -> None:
        if min(self.final_sign_weight, self.history_sign_weight) < 0.0:
            raise ValueError("v4.17 loss weights must be non-negative")
        if self.final_sign_weight + self.history_sign_weight <= 0.0:
            raise ValueError("v4.17 needs a non-zero sign loss")


@dataclass
class ObjectEventV417Loss:
    total: torch.Tensor
    final_sign: torch.Tensor
    history_sign: torch.Tensor


def signed_anchor_features(
    signed_anchor: torch.Tensor,
    *,
    train_scale: float,
    clip: float,
) -> torch.Tensor:
    """Return two scale-stable odd features from a signed event-only anchor."""
    if signed_anchor.ndim != 1:
        raise ValueError("signed_anchor must be [N]")
    if train_scale <= 0.0 or clip <= 0.0:
        raise ValueError("train_scale and clip must be positive")
    z = (signed_anchor / float(train_scale)).clamp(-float(clip), float(clip))
    return torch.stack((z, torch.tanh(z)), dim=1)


def signed_anchor_logits(
    signed_anchor: torch.Tensor,
    *,
    train_scale: float,
    clip: float,
    strength: float,
) -> torch.Tensor:
    """Map signed expansion to an exact-odd negative-class anchor logit.

    Positive expansion means approaching/positive class, therefore its BCE
    negative-class logit must be negative. Negative expansion produces a
    positive logit. The scale is estimated on train only.
    """
    if signed_anchor.ndim != 1:
        raise ValueError("signed_anchor must be [N]")
    if min(train_scale, clip, strength) <= 0.0:
        raise ValueError("anchor controls must be positive")
    z = (signed_anchor / float(train_scale)).clamp(-float(clip), float(clip))
    return -float(strength) * z


def temporal_sign_loss(
    *,
    sign_logit: torch.Tensor,
    instant_sign_logits: torch.Tensor,
    target_expansion: torch.Tensor,
    history_target_expansion: torch.Tensor,
    history_mask: torch.Tensor,
    config: ObjectEventV417LossConfig,
) -> ObjectEventV417Loss:
    """Uniform-prior sign loss; no sequence/sign importance reweighting."""
    if sign_logit.shape != target_expansion.shape:
        raise ValueError("current sign output and target must align")
    if instant_sign_logits.shape != history_target_expansion.shape:
        raise ValueError("history logits and targets must align")
    if history_mask.shape != instant_sign_logits.shape or history_mask.dtype != torch.bool:
        raise ValueError("history mask mismatch")

    negative = (target_expansion < 0.0).to(sign_logit.dtype)
    final_sign = functional.binary_cross_entropy_with_logits(sign_logit, negative)

    history_negative = (history_target_expansion < 0.0).to(instant_sign_logits.dtype)
    history = functional.binary_cross_entropy_with_logits(
        instant_sign_logits, history_negative, reduction="none"
    )
    valid = history_mask.to(history.dtype)
    history_sign = (history * valid).sum() / valid.sum().clamp_min(1.0)

    total = (
        config.final_sign_weight * final_sign
        + config.history_sign_weight * history_sign
    )
    return ObjectEventV417Loss(
        total=total,
        final_sign=final_sign,
        history_sign=history_sign,
    )


def v417_screen_gates(
    *,
    oof: Mapping[str, float],
    validation: Mapping[str, float],
    baseline: Mapping[str, float],
    anchor: Mapping[str, float],
    diagnostics: Mapping[str, float],
    gates: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "oof_balanced_sign": oof["balanced_sign_accuracy"] >= gates["oof_balanced_sign_gate"],
        "oof_negative_accuracy": oof["negative_accuracy"] >= gates["oof_negative_accuracy_gate"],
        "oof_min_sequence_negative_accuracy": oof["minimum_sequence_negative_accuracy"] >= gates["oof_min_sequence_negative_accuracy_gate"],
        "validation_pearson": validation["pearson"] >= gates["validation_pearson_gate"],
        "validation_expansion_mae": validation["expansion_mae"] <= gates["validation_expansion_mae_gate"],
        "validation_weighted_mid": validation["weighted_mid"] <= gates["validation_weighted_mid_gate"],
        "validation_weighted_rte_relative": validation["weighted_rte_percent"] <= gates["validation_weighted_rte_relative_ceiling"] * baseline["weighted_rte_percent"],
        "validation_saturation": validation["ttc_saturation_rate"] <= baseline["ttc_saturation_rate"] + gates["validation_saturation_max_increase"],
        "validation_balanced_sign": validation["balanced_sign_accuracy"] >= gates["validation_balanced_sign_gate"],
        "validation_negative_accuracy": validation["negative_accuracy"] >= gates["validation_negative_accuracy_gate"],
        "validation_min_sequence_pearson": validation["minimum_sequence_pearson"] >= gates["validation_min_sequence_pearson_gate"],
        "validation_min_sequence_negative_accuracy": validation["minimum_sequence_negative_accuracy"] >= gates["validation_min_sequence_negative_accuracy_gate"],
        "validation_positive_accuracy": validation["positive_accuracy"] >= gates["validation_positive_accuracy_gate"],
        "anchor_not_worse_than_v416_magnitude": anchor["magnitude_mae"] <= gates["anchor_magnitude_mae_gate"],
        "zero_event_dependence": diagnostics["zero_event_pearson_drop"] >= gates["zero_event_pearson_drop_gate"],
        "shuffled_event_dependence": diagnostics["shuffled_event_pearson_drop"] >= gates["shuffled_event_pearson_drop_gate"],
        "exact_sign_oddness": diagnostics["sign_oddness_max_abs"] <= gates["exact_sign_oddness_ceiling"],
    }


__all__ = [
    "ObjectEventV417Loss",
    "ObjectEventV417LossConfig",
    "signed_anchor_features",
    "signed_anchor_logits",
    "temporal_sign_loss",
    "v417_screen_gates",
]
