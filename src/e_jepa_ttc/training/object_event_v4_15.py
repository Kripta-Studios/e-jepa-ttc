"""Training and evaluation utilities for Object Event TTC v4.15."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class V415ThresholdSelection:
    tail_alpha: float
    threshold: float
    objective: float
    balanced_sign_accuracy: float
    positive_accuracy: float
    negative_accuracy: float
    minimum_sequence_negative_accuracy: float
    override_rate: float
    feasible: bool


def projected_numpy(
    baseline: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    magnitude_ceiling: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if baseline.shape != probability.shape or baseline.ndim != 1:
        raise ValueError("baseline and probability must be aligned vectors")
    if not np.isfinite(baseline).all() or not np.isfinite(probability).all():
        raise ValueError("projection inputs must be finite")
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must lie in [0.5,1)")
    if magnitude_ceiling is not None and (
        not np.isfinite(magnitude_ceiling) or magnitude_ceiling <= 0.0
    ):
        raise ValueError("magnitude_ceiling must be positive and finite")
    override = (baseline >= 0.0) & (probability >= threshold)
    if magnitude_ceiling is not None:
        override &= np.abs(baseline) <= float(magnitude_ceiling)
    prediction = baseline.copy()
    prediction[override] = -np.abs(prediction[override])
    return prediction, override


def sign_statistics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    minimum_negatives: int,
) -> dict[str, float]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    negative = target < 0.0
    positive = ~negative
    pos_acc = float(np.mean(prediction[positive] >= 0.0)) if positive.any() else 0.0
    neg_acc = float(np.mean(prediction[negative] < 0.0)) if negative.any() else 0.0
    minimum = 1.0
    eligible = 0
    for _, group in frame.assign(_prediction=prediction).groupby("sequence_id", sort=True):
        y = group["target_expansion"].to_numpy(dtype=np.float64)
        p = group["_prediction"].to_numpy(dtype=np.float64)
        mask = y < 0.0
        if int(mask.sum()) >= minimum_negatives:
            minimum = min(minimum, float(np.mean(p[mask] < 0.0)))
            eligible += 1
    if eligible == 0:
        minimum = 0.0
    return {
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_sign_accuracy": 0.5 * (pos_acc + neg_acc),
        "minimum_sequence_negative_accuracy": minimum,
    }



def positive_magnitude_ceiling(
    positive_magnitude: np.ndarray, quantile: float
) -> float:
    values = np.asarray(positive_magnitude, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("positive_magnitude must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("positive_magnitude must be finite and non-negative")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0,1)")
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, quantile, interpolation="higher"))


def positive_tail_threshold(positive_probability: np.ndarray, tail_alpha: float) -> float:
    values = np.asarray(positive_probability, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("positive_probability must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("positive_probability must be finite")
    if not 0.0 < tail_alpha < 0.5:
        raise ValueError("tail_alpha must lie in (0,0.5)")
    try:
        return float(np.quantile(values, 1.0 - tail_alpha, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, 1.0 - tail_alpha, interpolation="higher"))

def select_oof_threshold(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    tail_alphas: Sequence[float],
    positive_accuracy_floor: float,
    override_rate_ceiling: float,
    minimum_negatives: int,
) -> tuple[V415ThresholdSelection, pd.DataFrame]:
    baseline = frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != target.shape or probability.ndim != 1:
        raise ValueError("probability must align with frame")
    positive = target >= 0.0
    if not positive.any() or positive.all():
        raise ValueError("OOF tail calibration requires both signs")
    rows: list[dict[str, float | bool]] = []
    selections: list[V415ThresholdSelection] = []
    for tail_alpha in tail_alphas:
        threshold = positive_tail_threshold(probability[positive], float(tail_alpha))
        head_negative = probability >= threshold
        direct_prediction = np.where(head_negative, -1.0, 1.0)
        stats = sign_statistics(
            frame, direct_prediction, minimum_negatives=minimum_negatives
        )
        override = (baseline >= 0.0) & head_negative
        override_rate = float(np.mean(override))
        feasible = (
            stats["positive_accuracy"] >= positive_accuracy_floor
            and override_rate <= override_rate_ceiling
        )
        # The baseline already has almost perfect train signs, so projection-based
        # OOF selection degenerates to "never override".  Select the positive-tail
        # false-positive budget from direct OOF sign discrimination instead.
        objective = (
            2.0 * stats["minimum_sequence_negative_accuracy"]
            + stats["negative_accuracy"]
            + stats["balanced_sign_accuracy"]
            + 0.25 * stats["positive_accuracy"]
            - 0.25 * override_rate
        )
        selection = V415ThresholdSelection(
            tail_alpha=float(tail_alpha),
            threshold=float(threshold),
            objective=float(objective),
            balanced_sign_accuracy=stats["balanced_sign_accuracy"],
            positive_accuracy=stats["positive_accuracy"],
            negative_accuracy=stats["negative_accuracy"],
            minimum_sequence_negative_accuracy=stats[
                "minimum_sequence_negative_accuracy"
            ],
            override_rate=override_rate,
            feasible=bool(feasible),
        )
        selections.append(selection)
        rows.append(selection.__dict__)
    feasible = [selection for selection in selections if selection.feasible]
    pool = feasible if feasible else selections
    best = max(pool, key=lambda item: (item.objective, -item.tail_alpha))
    return best, pd.DataFrame.from_records(rows)


def v415_screen_gates(
    *,
    oof: Mapping[str, float],
    validation: Mapping[str, float],
    baseline: Mapping[str, float],
    diagnostics: Mapping[str, float],
    gates: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "oof_balanced_sign": oof["balanced_sign_accuracy"]
        >= gates["oof_balanced_sign_gate"],
        "oof_negative_accuracy": oof["negative_accuracy"]
        >= gates["oof_negative_accuracy_gate"],
        "oof_min_sequence_negative_accuracy": oof[
            "minimum_sequence_negative_accuracy"
        ]
        >= gates["oof_min_sequence_negative_accuracy_gate"],
        "validation_pearson": validation["pearson"]
        >= gates["validation_pearson_gate"],
        "validation_expansion_mae": validation["expansion_mae"]
        <= gates["validation_expansion_mae_gate"],
        "validation_weighted_mid": validation["weighted_mid"]
        <= gates["validation_weighted_mid_gate"],
        "validation_weighted_rte_relative": validation["weighted_rte_percent"]
        <= gates["validation_weighted_rte_relative_ceiling"]
        * baseline["weighted_rte_percent"],
        "validation_saturation": validation["ttc_saturation_rate"]
        <= baseline["ttc_saturation_rate"]
        + gates["validation_saturation_max_increase"],
        "validation_balanced_sign": validation["balanced_sign_accuracy"]
        >= gates["validation_balanced_sign_gate"],
        "validation_negative_accuracy": validation["negative_accuracy"]
        >= gates["validation_negative_accuracy_gate"],
        "validation_min_sequence_pearson": validation["minimum_sequence_pearson"]
        >= gates["validation_min_sequence_pearson_gate"],
        "validation_min_sequence_negative_accuracy": validation[
            "minimum_sequence_negative_accuracy"
        ]
        >= gates["validation_min_sequence_negative_accuracy_gate"],
        "validation_positive_accuracy": validation["positive_accuracy"]
        >= gates["validation_positive_accuracy_gate"],
        "validation_override_rate": diagnostics["override_rate"]
        <= gates["validation_override_rate_ceiling"],
        "zero_event_dependence": diagnostics["zero_event_pearson_drop"]
        >= gates["zero_event_pearson_drop_gate"],
        "shuffled_event_dependence": diagnostics["shuffled_event_pearson_drop"]
        >= gates["shuffled_event_pearson_drop_gate"],
    }


__all__ = [
    "V415ThresholdSelection",
    "positive_magnitude_ceiling",
    "positive_tail_threshold",
    "projected_numpy",
    "select_oof_threshold",
    "sign_statistics",
    "v415_screen_gates",
]
