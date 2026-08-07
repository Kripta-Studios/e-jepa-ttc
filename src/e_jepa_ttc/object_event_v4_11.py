"""Train-only sign/magnitude routing utilities for Object Event TTC v4.11.

The v4.10 fixed ensemble is globally stable but can be systematically wrong on
negative TTC windows when every expert predicts the same positive sign.  A
convex fusion cannot repair that failure.  V4.11 therefore learns a small,
seed-invariant sign head from the development *train* predictions only and uses
it to apply a bounded non-convex negative residual to the fixed v4.10 magnitude.

Sequence identifiers, track identifiers, boxes, visible heights, validation
labels, official eAP test labels, and EvTTC labels are never model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from e_jepa_ttc.object_event_v4_4 import branch_metrics, pearson, sequence_sign_weights

IDENTITY_COLUMNS: tuple[str, ...] = ("sequence_id", "sample_token", "track_id")
TARGET_COLUMNS: tuple[str, ...] = ("delta_t_s", "target_ttc_s", "target_expansion")
BRANCH_COLUMNS: Mapping[str, tuple[str, str, str]] = {
    "event": (
        "base_prediction_expansion",
        "dense_prediction_expansion",
        "fused_prediction_expansion",
    ),
    "zero": (
        "base_zero_events_expansion",
        "dense_zero_events_expansion",
        "fused_zero_events_expansion",
    ),
    "shuffled": (
        "base_shuffled_mean_expansion",
        "dense_shuffled_mean_expansion",
        "fused_shuffled_mean_expansion",
    ),
}
REQUIRED_V49_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *TARGET_COLUMNS,
    *(column for columns in BRANCH_COLUMNS.values() for column in columns),
)
FEATURE_NAMES: tuple[str, ...] = (
    "base_mean",
    "dense_mean",
    "fixed_mean",
    "base_std",
    "dense_std",
    "fixed_std",
    "base_min",
    "dense_min",
    "fixed_min",
    "base_max",
    "dense_max",
    "fixed_max",
    "dense_minus_base_mean",
    "dense_minus_base_std",
    "expert_abs_gap_mean",
    "expert_abs_gap_max",
    "base_negative_vote",
    "dense_negative_vote",
    "fixed_negative_vote",
    "base_dense_product_mean",
    "fixed_abs_mean",
    "fixed_signed_sqrt",
)


@dataclass(frozen=True)
class V411RouterConfig:
    seeds: tuple[int, ...] = (7, 13, 23)
    l2_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    negative_threshold_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    flip_scale_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    maximum_iterations: int = 80
    convergence_tolerance: float = 1.0e-8
    hessian_damping: float = 1.0e-6
    minimum_flip_magnitude: float = 0.002
    per_sequence_negative_min_count: int = 20
    max_abs_expansion: float = 0.25
    ttc_clip_seconds: float = 60.0
    minimum_abs_expansion: float = 1.0e-4
    cv_pearson_floor: float = 0.88
    cv_expansion_mae_ceiling: float = 0.015
    validation_pearson_floor: float = 0.64
    validation_pearson_max_drop: float = 0.02
    validation_expansion_mae_tolerance: float = 0.001
    validation_balanced_sign_floor: float = 0.75
    validation_negative_accuracy_floor: float = 0.65
    validation_min_sequence_pearson_floor: float = 0.50
    validation_min_sequence_negative_accuracy_floor: float = 0.30
    validation_track_bootstrap_lower_floor: float = 0.55
    zero_event_pearson_drop_floor: float = 0.55
    shuffled_event_pearson_drop_floor: float = 0.55
    track_bootstrap_repeats: int = 3000

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.11 requires at least three unique seeds")
        if not self.l2_grid or min(self.l2_grid) <= 0.0:
            raise ValueError("l2_grid values must be positive")
        if not self.negative_threshold_grid or not all(
            0.0 < value < 1.0 for value in self.negative_threshold_grid
        ):
            raise ValueError("negative thresholds must lie in (0,1)")
        if not self.flip_scale_grid or not all(0.0 < value <= 1.0 for value in self.flip_scale_grid):
            raise ValueError("flip scales must lie in (0,1]")
        if self.maximum_iterations < 5:
            raise ValueError("maximum_iterations must be at least 5")
        if min(
            self.convergence_tolerance,
            self.hessian_damping,
            self.minimum_flip_magnitude,
            self.minimum_abs_expansion,
        ) <= 0.0:
            raise ValueError("v4.11 numerical controls must be positive")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")
        if self.track_bootstrap_repeats < 100:
            raise ValueError("track_bootstrap_repeats must be at least 100")


@dataclass(frozen=True)
class WeightedLogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    l2: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict_negative_probability(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("logistic feature dimension mismatch")
        standardized = (values - self.mean) / self.scale
        design = np.column_stack((np.ones(len(values), dtype=np.float64), standardized))
        logits = np.clip(design @ self.coefficients, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_jsonable(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "l2": float(self.l2),
        }


def _require_v49_columns(frame: pd.DataFrame, *, name: str) -> None:
    missing = [column for column in REQUIRED_V49_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate identities")


def align_seed_experts(
    seed_frames: Mapping[int, pd.DataFrame],
    *,
    split_name: str,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Align v4.9 expert outputs by identity while preserving each seed."""

    selected = tuple(sorted(seed_frames)) if seeds is None else tuple(int(seed) for seed in seeds)
    if len(selected) < 2:
        raise ValueError("at least two seed frames are required")
    if set(selected) != set(seed_frames):
        raise ValueError("seed_frames and requested seeds differ")

    normalized: dict[int, pd.DataFrame] = {}
    for seed in selected:
        frame = seed_frames[seed]
        _require_v49_columns(frame, name=f"{split_name} seed {seed}")
        normalized[seed] = frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
            drop=True
        )

    first = normalized[selected[0]]
    result = first.loc[:, [*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
    for seed in selected:
        frame = normalized[seed]
        if not result.loc[:, list(IDENTITY_COLUMNS)].equals(frame.loc[:, list(IDENTITY_COLUMNS)]):
            raise ValueError(f"{split_name} identities do not align for seed {seed}")
        for column in TARGET_COLUMNS:
            if not np.allclose(
                result[column].to_numpy(dtype=np.float64),
                frame[column].to_numpy(dtype=np.float64),
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError(f"{split_name} target mismatch in {column} for seed {seed}")
        for branch, columns in BRANCH_COLUMNS.items():
            for expert, column in zip(("base", "dense", "fixed"), columns, strict=True):
                result[f"{branch}_{expert}_seed_{seed}"] = frame[column].to_numpy(
                    dtype=np.float64
                )
    return result


def _seed_matrix(frame: pd.DataFrame, *, branch: str, expert: str, seeds: Sequence[int]) -> np.ndarray:
    columns = [f"{branch}_{expert}_seed_{int(seed)}" for seed in seeds]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing aligned expert columns: {missing}")
    return frame[columns].to_numpy(dtype=np.float64)


def router_features(
    frame: pd.DataFrame,
    *,
    branch: str,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return seed-invariant router features and the fixed ensemble magnitude."""

    if branch not in BRANCH_COLUMNS:
        raise KeyError(branch)
    base = _seed_matrix(frame, branch=branch, expert="base", seeds=seeds)
    dense = _seed_matrix(frame, branch=branch, expert="dense", seeds=seeds)
    fixed = _seed_matrix(frame, branch=branch, expert="fixed", seeds=seeds)
    gap = dense - base
    fixed_mean = fixed.mean(axis=1)
    features = np.column_stack(
        (
            base.mean(axis=1),
            dense.mean(axis=1),
            fixed_mean,
            base.std(axis=1, ddof=0),
            dense.std(axis=1, ddof=0),
            fixed.std(axis=1, ddof=0),
            base.min(axis=1),
            dense.min(axis=1),
            fixed.min(axis=1),
            base.max(axis=1),
            dense.max(axis=1),
            fixed.max(axis=1),
            gap.mean(axis=1),
            gap.std(axis=1, ddof=0),
            np.abs(gap).mean(axis=1),
            np.abs(gap).max(axis=1),
            (base < 0.0).mean(axis=1),
            (dense < 0.0).mean(axis=1),
            (fixed < 0.0).mean(axis=1),
            (base * dense).mean(axis=1),
            np.abs(fixed).mean(axis=1),
            np.sign(fixed_mean) * np.sqrt(np.abs(fixed_mean)),
        )
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("unexpected v4.11 feature count")
    return np.nan_to_num(features), fixed_mean


def fit_weighted_logistic(
    features: np.ndarray,
    negative_target: np.ndarray,
    *,
    l2: float,
    sample_weight: np.ndarray | None = None,
    maximum_iterations: int = 80,
    convergence_tolerance: float = 1.0e-8,
    hessian_damping: float = 1.0e-6,
) -> WeightedLogisticModel:
    """Fit a standardized, class-balanced L2 logistic model using damped IRLS."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(negative_target, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.shape[0] or len(target) < 4:
        raise ValueError("invalid logistic training shapes")
    if not np.isin(target, (0.0, 1.0)).all() or np.unique(target).size != 2:
        raise ValueError("negative_target must contain both binary classes")
    if l2 <= 0.0:
        raise ValueError("l2 must be positive")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")

    if sample_weight is None:
        weight = np.ones(len(target), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weight.shape != target.shape or np.any(weight <= 0.0) or not np.isfinite(weight).all():
            raise ValueError("sample_weight must be finite and positive")
    weight = weight / max(float(np.mean(weight)), 1.0e-12)

    mean = np.average(values, axis=0, weights=weight)
    centered = values - mean
    variance = np.average(np.square(centered), axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 1.0e-12))
    standardized = centered / scale
    design = np.column_stack((np.ones(len(values), dtype=np.float64), standardized))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.ones(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0] = 0.0

    for _ in range(int(maximum_iterations)):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (weight * (probability - target)) / weight.sum()
        gradient += penalty * coefficients
        curvature = weight * np.maximum(probability * (1.0 - probability), 1.0e-6)
        hessian = (design.T * curvature) @ design / weight.sum()
        hessian += np.diag(penalty + float(hessian_damping))
        hessian[0, 0] -= float(hessian_damping)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.linalg.norm(step)) <= convergence_tolerance * (
            1.0 + float(np.linalg.norm(coefficients))
        ):
            break

    return WeightedLogisticModel(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        l2=float(l2),
    )


def apply_negative_repair(
    fixed_prediction: np.ndarray,
    negative_probability: np.ndarray,
    *,
    threshold: float,
    flip_scale: float,
    minimum_flip_magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip only high-confidence false-positive candidates to bounded negatives."""

    fixed = np.asarray(fixed_prediction, dtype=np.float64).reshape(-1)
    probability = np.asarray(negative_probability, dtype=np.float64).reshape(-1)
    if fixed.shape != probability.shape:
        raise ValueError("repair shapes must match")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0,1)")
    if not 0.0 < flip_scale <= 1.0:
        raise ValueError("flip_scale must lie in (0,1]")
    if minimum_flip_magnitude <= 0.0:
        raise ValueError("minimum_flip_magnitude must be positive")
    repair_mask = (fixed >= 0.0) & (probability >= float(threshold))
    output = fixed.copy()
    magnitude = np.maximum(np.abs(fixed), float(minimum_flip_magnitude))
    output[repair_mask] = -float(flip_scale) * magnitude[repair_mask]
    return output, repair_mask


def _per_sequence_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    config: V411RouterConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(frame) != len(prediction):
        raise ValueError("prediction length mismatch")
    for sequence_id, indices in frame.groupby("sequence_id", sort=True).indices.items():
        index = np.asarray(indices, dtype=np.int64)
        metrics = branch_metrics(
            frame.iloc[index]["target_expansion"].to_numpy(dtype=np.float64),
            prediction[index],
            frame.iloc[index]["delta_t_s"].to_numpy(dtype=np.float64),
            ttc_clip_seconds=config.ttc_clip_seconds,
            minimum_abs_expansion=config.minimum_abs_expansion,
        )
        rows.append({"sequence_id": str(sequence_id), **metrics})
    return pd.DataFrame.from_records(rows)


def routing_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    config: V411RouterConfig,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    metrics: dict[str, float | int] = branch_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        ttc_clip_seconds=config.ttc_clip_seconds,
        minimum_abs_expansion=config.minimum_abs_expansion,
    )
    per_sequence = _per_sequence_metrics(frame, prediction, config=config)
    eligible = per_sequence[
        per_sequence["negative_count"] >= config.per_sequence_negative_min_count
    ]
    metrics["sequence_macro_pearson"] = float(per_sequence["pearson"].mean())
    metrics["minimum_sequence_pearson"] = float(per_sequence["pearson"].min())
    metrics["minimum_sequence_negative_accuracy"] = (
        float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    )
    metrics["eligible_negative_sequence_count"] = int(len(eligible))
    return metrics, per_sequence


def _candidate_key(record: Mapping[str, float | int | bool]) -> tuple[float, ...]:
    return (
        float(bool(record["feasible"])),
        float(record["minimum_sequence_negative_accuracy"]),
        float(record["negative_accuracy"]),
        float(record["balanced_sign_accuracy"]),
        float(record["pearson"]),
        -float(record["expansion_mae"]),
        -float(record["repair_fraction"]),
    )


def select_router_train_only(
    train_frame: pd.DataFrame,
    *,
    config: V411RouterConfig,
) -> tuple[WeightedLogisticModel, dict[str, float | int | bool], pd.DataFrame, pd.DataFrame]:
    """Select the router by leave-one-train-sequence-out predictions only."""

    features, fixed = router_features(train_frame, branch="event", seeds=config.seeds)
    target = train_frame["target_expansion"].to_numpy(dtype=np.float64)
    negative = (target < 0.0).astype(np.float64)
    sequences = train_frame["sequence_id"].astype(str).to_numpy(dtype=object)
    unique_sequences = tuple(sorted(set(sequences.tolist())))
    if len(unique_sequences) < 3:
        raise ValueError("v4.11 requires at least three train sequences for grouped CV")

    base_weight = sequence_sign_weights(sequences, target, cap=10.0)
    oof_probability_by_l2: dict[float, np.ndarray] = {}
    for l2 in config.l2_grid:
        oof = np.full(len(train_frame), np.nan, dtype=np.float64)
        for held_out in unique_sequences:
            validation_mask = sequences == held_out
            training_mask = ~validation_mask
            model = fit_weighted_logistic(
                features[training_mask],
                negative[training_mask],
                l2=float(l2),
                sample_weight=base_weight[training_mask],
                maximum_iterations=config.maximum_iterations,
                convergence_tolerance=config.convergence_tolerance,
                hessian_damping=config.hessian_damping,
            )
            oof[validation_mask] = model.predict_negative_probability(features[validation_mask])
        if not np.isfinite(oof).all():
            raise RuntimeError(f"incomplete grouped predictions for l2={l2}")
        oof_probability_by_l2[float(l2)] = oof

    records: list[dict[str, float | int | bool]] = []
    predictions: dict[tuple[float, float, float], np.ndarray] = {}
    for l2, probability in oof_probability_by_l2.items():
        for threshold in config.negative_threshold_grid:
            for flip_scale in config.flip_scale_grid:
                routed, repaired = apply_negative_repair(
                    fixed,
                    probability,
                    threshold=float(threshold),
                    flip_scale=float(flip_scale),
                    minimum_flip_magnitude=config.minimum_flip_magnitude,
                )
                metrics, _ = routing_metrics(train_frame, routed, config=config)
                record: dict[str, float | int | bool] = {
                    "l2": float(l2),
                    "negative_threshold": float(threshold),
                    "flip_scale": float(flip_scale),
                    **metrics,
                    "repair_count": int(repaired.sum()),
                    "repair_fraction": float(repaired.mean()),
                    "feasible": bool(
                        float(metrics["pearson"]) >= config.cv_pearson_floor
                        and float(metrics["expansion_mae"]) <= config.cv_expansion_mae_ceiling
                    ),
                }
                records.append(record)
                predictions[(float(l2), float(threshold), float(flip_scale))] = routed

    candidates = pd.DataFrame.from_records(records)
    best = max(records, key=_candidate_key)
    key = (
        float(best["l2"]),
        float(best["negative_threshold"]),
        float(best["flip_scale"]),
    )
    selected_prediction = predictions[key]
    selected_probability = oof_probability_by_l2[float(best["l2"])]
    final_model = fit_weighted_logistic(
        features,
        negative,
        l2=float(best["l2"]),
        sample_weight=base_weight,
        maximum_iterations=config.maximum_iterations,
        convergence_tolerance=config.convergence_tolerance,
        hessian_damping=config.hessian_damping,
    )
    oof_rows = train_frame.loc[:, [*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
    oof_rows["fixed_prediction_expansion"] = fixed
    oof_rows["negative_probability"] = selected_probability
    oof_rows["routed_prediction_expansion"] = selected_prediction
    oof_rows["negative_repair"] = (
        (fixed >= 0.0) & (selected_probability >= float(best["negative_threshold"]))
    )
    return final_model, best, candidates, oof_rows


def validation_gates(
    *,
    baseline: Mapping[str, object],
    routed: Mapping[str, object],
    dependence: Mapping[str, float],
    bootstrap_lower: float,
    selection_feasible: bool,
    config: V411RouterConfig,
) -> dict[str, bool]:
    return {
        "train_only_selection_feasible": bool(selection_feasible),
        "pearson_floor": float(routed["pearson"]) >= config.validation_pearson_floor,
        "pearson_preserved": float(routed["pearson"])
        >= float(baseline["pearson"]) - config.validation_pearson_max_drop,
        "expansion_mae": float(routed["expansion_mae"])
        <= float(baseline["expansion_mae"]) + config.validation_expansion_mae_tolerance,
        "balanced_sign": float(routed["balanced_sign_accuracy"])
        >= config.validation_balanced_sign_floor,
        "negative_accuracy": float(routed["negative_accuracy"])
        >= config.validation_negative_accuracy_floor,
        "minimum_sequence_pearson": float(routed["minimum_sequence_pearson"])
        >= config.validation_min_sequence_pearson_floor,
        "minimum_sequence_negative_accuracy": float(
            routed["minimum_sequence_negative_accuracy"]
        )
        >= config.validation_min_sequence_negative_accuracy_floor,
        "track_bootstrap_lower": float(bootstrap_lower)
        >= config.validation_track_bootstrap_lower_floor,
        "zero_event_dependence": float(dependence["zero_event_pearson_drop"])
        >= config.zero_event_pearson_drop_floor,
        "shuffled_event_dependence": float(dependence["shuffled_event_pearson_drop"])
        >= config.shuffled_event_pearson_drop_floor,
    }


__all__ = [
    "BRANCH_COLUMNS",
    "FEATURE_NAMES",
    "IDENTITY_COLUMNS",
    "REQUIRED_V49_COLUMNS",
    "TARGET_COLUMNS",
    "V411RouterConfig",
    "WeightedLogisticModel",
    "align_seed_experts",
    "apply_negative_repair",
    "fit_weighted_logistic",
    "router_features",
    "routing_metrics",
    "select_router_train_only",
    "validation_gates",
]
