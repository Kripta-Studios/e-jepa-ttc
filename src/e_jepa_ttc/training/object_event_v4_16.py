"""Window construction, losses and gates for Object Event TTC v4.16."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional


@dataclass(frozen=True)
class ObjectEventV416LossConfig:
    final_sign_weight: float = 1.0
    history_sign_weight: float = 0.35
    magnitude_target_weight: float = 1.0
    magnitude_teacher_weight: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.final_sign_weight,
            self.history_sign_weight,
            self.magnitude_target_weight,
            self.magnitude_teacher_weight,
        )
        if min(values) < 0.0 or sum(values) <= 0.0:
            raise ValueError("v4.16 loss weights must be non-negative and non-zero")


@dataclass
class ObjectEventV416Loss:
    total: torch.Tensor
    final_sign: torch.Tensor
    history_sign: torch.Tensor
    magnitude_target: torch.Tensor
    magnitude_teacher: torch.Tensor


def _timestamp_from_token(token: str) -> int:
    try:
        return int(str(token).rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"sample_token lacks numeric timestamp: {token!r}") from exc


def causal_window_indices(
    frame: pd.DataFrame,
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-aligned chronological histories inside each track.

    Returned rows remain aligned with the original frame. sequence_id/track_id
    are grouping metadata only and never become model features.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    required = {"sequence_id", "track_id", "sample_token"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing columns: {missing}")
    count = len(frame)
    index = np.full((count, window_size), -1, dtype=np.int64)
    mask = np.zeros((count, window_size), dtype=bool)
    history_length = np.zeros(count, dtype=np.int64)
    work = frame.loc[:, ["sequence_id", "track_id", "sample_token"]].copy()
    work["_row"] = np.arange(count, dtype=np.int64)
    work["_timestamp"] = [_timestamp_from_token(v) for v in work["sample_token"]]
    for _, group in work.groupby(["sequence_id", "track_id"], sort=False):
        ordered = group.sort_values(["_timestamp", "_row"], kind="stable")["_row"].to_numpy(dtype=np.int64)
        for position, row in enumerate(ordered):
            history = ordered[max(0, position - window_size + 1) : position + 1]
            start = window_size - len(history)
            index[row, start:] = history
            mask[row, start:] = True
            history_length[row] = len(history)
    if not np.all(mask[:, -1]):
        raise AssertionError("current sample must occupy the final causal slot")
    return index, mask, history_length


def gather_windows(values: torch.Tensor, indices: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must be [N,D]")
    if indices.shape != mask.shape or indices.ndim != 2:
        raise ValueError("indices and mask must share [N,T]")
    if len(values) != len(indices):
        raise ValueError("window rows must align with values")
    safe = np.maximum(indices, 0)
    gathered = values[torch.as_tensor(safe, dtype=torch.long)].clone()
    gathered[~torch.as_tensor(mask, dtype=torch.bool)] = 0.0
    return gathered


def gather_scalar_windows(values: torch.Tensor, indices: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    if values.ndim != 1 or len(values) != len(indices):
        raise ValueError("scalar values must align with windows")
    safe = np.maximum(indices, 0)
    gathered = values[torch.as_tensor(safe, dtype=torch.long)].clone()
    gathered[~torch.as_tensor(mask, dtype=torch.bool)] = 0.0
    return gathered



def uniform_epoch_indices(
    train_indices: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Visit each selected row exactly once in a random order.

    Sequence/sign importance weights are already applied inside the v4.16 loss.
    Sampling with the same weights as well would square their effective leverage
    and over-emphasize rare sequence/sign cells.
    """
    if train_indices.ndim != 1:
        raise ValueError("train_indices must be one-dimensional")
    if len(train_indices) == 0:
        raise ValueError("train_indices must be non-empty")
    order = torch.randperm(len(train_indices), generator=generator)
    return train_indices[order]

def temporal_dual_head_loss(
    *,
    sign_logit: torch.Tensor,
    instant_sign_logits: torch.Tensor,
    magnitude: torch.Tensor,
    target_expansion: torch.Tensor,
    teacher_magnitude: torch.Tensor,
    history_target_expansion: torch.Tensor,
    history_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    magnitude_floor: float,
    config: ObjectEventV416LossConfig,
) -> ObjectEventV416Loss:
    if sign_logit.shape != target_expansion.shape or magnitude.shape != target_expansion.shape:
        raise ValueError("current outputs and targets must align")
    if instant_sign_logits.shape != history_target_expansion.shape:
        raise ValueError("history logits and targets must align")
    if history_mask.shape != instant_sign_logits.shape or history_mask.dtype != torch.bool:
        raise ValueError("history mask mismatch")
    if sample_weight.shape != target_expansion.shape:
        raise ValueError("sample weights must align")
    if magnitude_floor <= 0.0:
        raise ValueError("magnitude_floor must be positive")

    negative = (target_expansion < 0.0).to(sign_logit.dtype)
    current_bce = functional.binary_cross_entropy_with_logits(sign_logit, negative, reduction="none")
    weight = sample_weight.to(current_bce.dtype).clamp_min(0.0)
    normaliser = weight.sum().clamp_min(1.0e-12)
    final_sign = (current_bce * weight).sum() / normaliser

    history_negative = (history_target_expansion < 0.0).to(instant_sign_logits.dtype)
    history_bce = functional.binary_cross_entropy_with_logits(
        instant_sign_logits, history_negative, reduction="none"
    )
    valid = history_mask.to(history_bce.dtype)
    history_sign = (history_bce * valid).sum() / valid.sum().clamp_min(1.0)

    target_magnitude = target_expansion.abs().clamp_min(magnitude_floor)
    teacher = teacher_magnitude.abs().clamp_min(magnitude_floor)
    log_prediction = torch.log(magnitude.clamp_min(magnitude_floor))
    magnitude_target = functional.smooth_l1_loss(
        log_prediction, torch.log(target_magnitude), reduction="none"
    )
    magnitude_teacher = functional.smooth_l1_loss(
        log_prediction, torch.log(teacher), reduction="none"
    )
    magnitude_target = (magnitude_target * weight).sum() / normaliser
    magnitude_teacher = (magnitude_teacher * weight).sum() / normaliser

    total = (
        config.final_sign_weight * final_sign
        + config.history_sign_weight * history_sign
        + config.magnitude_target_weight * magnitude_target
        + config.magnitude_teacher_weight * magnitude_teacher
    )
    return ObjectEventV416Loss(
        total=total,
        final_sign=final_sign,
        history_sign=history_sign,
        magnitude_target=magnitude_target,
        magnitude_teacher=magnitude_teacher,
    )


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


def v416_screen_gates(
    *,
    oof: Mapping[str, float],
    validation: Mapping[str, float],
    baseline: Mapping[str, float],
    diagnostics: Mapping[str, float],
    gates: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "oof_balanced_sign": oof["balanced_sign_accuracy"] >= gates["oof_balanced_sign_gate"],
        "oof_negative_accuracy": oof["negative_accuracy"] >= gates["oof_negative_accuracy_gate"],
        "oof_min_sequence_negative_accuracy": oof["minimum_sequence_negative_accuracy"] >= gates["oof_min_sequence_negative_accuracy_gate"],
        "oof_expansion_mae": oof["expansion_mae"] <= gates["oof_expansion_mae_gate"],
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
        "zero_event_dependence": diagnostics["zero_event_pearson_drop"] >= gates["zero_event_pearson_drop_gate"],
        "shuffled_event_dependence": diagnostics["shuffled_event_pearson_drop"] >= gates["shuffled_event_pearson_drop_gate"],
        "exact_sign_oddness": diagnostics["sign_oddness_max_abs"] <= gates["exact_sign_oddness_ceiling"],
        "exact_magnitude_evenness": diagnostics["magnitude_evenness_max_abs"] <= gates["exact_magnitude_evenness_ceiling"],
    }


__all__ = [
    "ObjectEventV416Loss",
    "ObjectEventV416LossConfig",
    "causal_window_indices",
    "gather_scalar_windows",
    "gather_windows",
    "sign_statistics",
    "temporal_dual_head_loss",
    "uniform_epoch_indices",
    "v416_screen_gates",
]
