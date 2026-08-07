"""V4.13 conservative dual-head fusion.

The v4.12 odd sign probe recovers receding motion but over-corrects positives.
V4.13 keeps the stable v4.10 signed magnitude as the default and injects a
soft directional score everywhere, while allowing a sign-changing correction
only for extremely confident receding predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ObjectEventV413Config:
    base_blend: float = 0.20
    override_blend: float = 0.51
    negative_override_probability: float = 0.985

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_blend < 0.5:
            raise ValueError("base_blend must lie in [0, 0.5)")
        if not 0.5 < self.override_blend <= 1.0:
            raise ValueError("override_blend must lie in (0.5, 1]")
        if not 0.5 < self.negative_override_probability < 1.0:
            raise ValueError("negative_override_probability must lie in (0.5, 1)")


def conservative_dual_head_prediction(
    baseline_prediction: np.ndarray,
    negative_probability: np.ndarray,
    *,
    config: ObjectEventV413Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return routed prediction, soft direction, blend and override mask.

    The directional score is continuous: |g| (1 - 2 p_negative).  A small
    fixed blend uses it as a residual without changing most signs.  Only a
    baseline-positive sample above the locked confidence threshold receives a
    blend greater than 0.5 and can therefore cross zero.
    """
    cfg = config or ObjectEventV413Config()
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    probability = np.asarray(negative_probability, dtype=np.float64)
    if baseline.shape != probability.shape:
        raise ValueError("baseline_prediction and negative_probability must align")
    if baseline.ndim != 1:
        raise ValueError("v4.13 expects one-dimensional prediction arrays")
    if not np.isfinite(baseline).all() or not np.isfinite(probability).all():
        raise ValueError("v4.13 inputs must be finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("negative_probability must lie in [0,1]")

    magnitude = np.abs(baseline)
    directional = magnitude * (1.0 - 2.0 * probability)
    override = (baseline >= 0.0) & (
        probability >= cfg.negative_override_probability
    )
    blend = np.full_like(baseline, cfg.base_blend)
    blend[override] = cfg.override_blend
    routed = (1.0 - blend) * baseline + blend * directional
    return routed, directional, blend, override


def selective_fusion_gates(
    *,
    routed: Mapping[str, float],
    baseline: Mapping[str, float],
    diagnostics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "pearson_floor": routed["pearson"] >= thresholds["pearson_floor"],
        "pearson_preserved": routed["pearson"]
        >= baseline["pearson"] - thresholds["pearson_max_drop"],
        "mae_preserved": routed["expansion_mae"]
        <= baseline["expansion_mae"] + thresholds["mae_tolerance"],
        "balanced_sign": routed["balanced_sign_accuracy"]
        >= thresholds["balanced_sign_gate"],
        "negative_accuracy": routed["negative_accuracy"]
        >= thresholds["negative_accuracy_gate"],
        "minimum_sequence_negative_accuracy": routed[
            "minimum_sequence_negative_accuracy"
        ]
        >= thresholds["minimum_sequence_negative_accuracy_gate"],
        "positive_accuracy_preserved": routed["positive_accuracy"]
        >= baseline["positive_accuracy"]
        - thresholds["positive_accuracy_max_drop"],
        "override_rate": diagnostics["override_rate"]
        <= thresholds["maximum_override_rate"],
        "zero_event_dependence": diagnostics["zero_event_pearson_drop"]
        >= thresholds["zero_event_pearson_drop_gate"],
        "shuffled_event_dependence": diagnostics["shuffled_event_pearson_drop"]
        >= thresholds["shuffled_event_pearson_drop_gate"],
    }


__all__ = [
    "ObjectEventV413Config",
    "conservative_dual_head_prediction",
    "selective_fusion_gates",
]
