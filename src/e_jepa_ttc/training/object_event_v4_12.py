"""Losses and gates for the v4.12 reversal-balanced directional sign probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ObjectEventV412LossConfig:
    original_bce_weight: float = 1.0
    reversed_bce_weight: float = 1.0
    antisymmetry_weight: float = 0.25
    margin_weight: float = 0.10
    margin: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.original_bce_weight,
            self.reversed_bce_weight,
            self.antisymmetry_weight,
            self.margin_weight,
            self.margin,
        ) < 0.0:
            raise ValueError("v4.12 loss weights must be non-negative")
        if self.original_bce_weight + self.reversed_bce_weight <= 0.0:
            raise ValueError("at least one BCE term must be active")


@dataclass
class ObjectEventV412LossOutput:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


def _balanced_bce(logits: torch.Tensor, negative_target: torch.Tensor) -> torch.Tensor:
    target = negative_target.to(dtype=logits.dtype)
    negative_count = target.sum().clamp_min(1.0)
    positive_count = (1.0 - target).sum().clamp_min(1.0)
    negative_weight = positive_count / negative_count
    weight = torch.where(target > 0.5, negative_weight, torch.ones_like(target))
    return functional.binary_cross_entropy_with_logits(logits, target, weight=weight)


def reversal_balanced_sign_loss(
    original_logits: torch.Tensor,
    reversed_logits: torch.Tensor,
    target_expansion: torch.Tensor,
    *,
    config: ObjectEventV412LossConfig | None = None,
) -> ObjectEventV412LossOutput:
    cfg = config or ObjectEventV412LossConfig()
    if original_logits.shape != reversed_logits.shape or original_logits.shape != target_expansion.shape:
        raise ValueError("v4.12 sign-loss tensors must have identical shapes")
    negative = target_expansion < 0.0
    reversed_negative = ~negative
    original_bce = _balanced_bce(original_logits, negative)
    reversed_bce = _balanced_bce(reversed_logits, reversed_negative)
    antisymmetry = (original_logits + reversed_logits).square().mean()
    signed_target = torch.where(
        negative,
        -torch.ones_like(original_logits),
        torch.ones_like(original_logits),
    )
    # A positive signed target means approach; negative means recede. The sign
    # logit is defined as probability of receding, hence -signed_target.
    margin = functional.relu(cfg.margin + signed_target * original_logits).mean()
    total = (
        cfg.original_bce_weight * original_bce
        + cfg.reversed_bce_weight * reversed_bce
        + cfg.antisymmetry_weight * antisymmetry
        + cfg.margin_weight * margin
    )
    return ObjectEventV412LossOutput(
        total=total,
        components={
            "original_bce": original_bce,
            "reversed_bce": reversed_bce,
            "antisymmetry": antisymmetry,
            "margin": margin,
        },
    )


def _screen_core_gates(
    *,
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "pearson_floor": metrics["pearson"] >= thresholds["screen_pearson_floor"],
        "pearson_preserved": metrics["pearson"]
        >= baseline["pearson"] - thresholds["screen_pearson_max_drop"],
        "mae_preserved": metrics["expansion_mae"]
        <= baseline["expansion_mae"] + thresholds["screen_mae_tolerance"],
        "balanced_sign": metrics["balanced_sign_accuracy"]
        >= thresholds["screen_balanced_sign_gate"],
        "negative_accuracy": metrics["negative_accuracy"]
        >= thresholds["screen_negative_accuracy_gate"],
        "minimum_sequence_negative_accuracy": metrics[
            "minimum_sequence_negative_accuracy"
        ]
        >= thresholds["screen_min_sequence_negative_accuracy_gate"],
        "reverse_accuracy": metrics["reverse_sign_accuracy"]
        >= thresholds["screen_reverse_accuracy_gate"],
    }


def directional_sign_checkpoint_gates(
    *,
    mode: str,
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    """Gates available during epoch-level checkpoint selection.

    Zero-event and shuffled-event dependence require additional full-dataset
    forward passes. They are intentionally evaluated only once for the selected
    checkpoint, not inside every screen epoch. The final scientific gates remain
    strict and still require both dependence metrics.
    """

    if mode == "overfit":
        return directional_sign_gates(
            mode=mode, metrics=metrics, baseline=baseline, thresholds=thresholds
        )
    if mode != "screen":
        raise ValueError("mode must be overfit or screen")
    return _screen_core_gates(
        metrics=metrics, baseline=baseline, thresholds=thresholds
    )


def directional_sign_gates(
    *,
    mode: str,
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    if mode not in {"overfit", "screen"}:
        raise ValueError("mode must be overfit or screen")
    if mode == "overfit":
        return {
            "balanced_sign": metrics["balanced_sign_accuracy"]
            >= thresholds["overfit_balanced_sign_gate"],
            "negative_accuracy": metrics["negative_accuracy"]
            >= thresholds["overfit_negative_accuracy_gate"],
            "reverse_accuracy": metrics["reverse_sign_accuracy"]
            >= thresholds["overfit_reverse_accuracy_gate"],
            "antisymmetry": metrics["antisymmetry_mean_abs"]
            <= thresholds["overfit_antisymmetry_ceiling"],
        }
    gates = _screen_core_gates(
        metrics=metrics, baseline=baseline, thresholds=thresholds
    )
    gates.update(
        {
            "zero_event_dependence": metrics["zero_event_pearson_drop"]
            >= thresholds["zero_event_pearson_drop_gate"],
            "shuffled_event_dependence": metrics["shuffled_event_pearson_drop"]
            >= thresholds["shuffled_event_pearson_drop_gate"],
        }
    )
    return gates


__all__ = [
    "ObjectEventV412LossConfig",
    "ObjectEventV412LossOutput",
    "directional_sign_checkpoint_gates",
    "directional_sign_gates",
    "reversal_balanced_sign_loss",
]
