"""Fixed development-fusion diagnostics for Object Event TTC v4.9.

V4.2 and v4.8 are complementary event-only experts.  V4.9 does not train a
new gate: it evaluates one fixed convex coefficient chosen before this v4.9 run and
keeps an alpha sweep as a diagnostic only.  The coefficient is informed by prior
development screens and is not suitable for a final benchmark claim.  The official eAP test and EvTTC stay
sealed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from e_jepa_ttc.object_event_v4_4 import branch_metrics, official_eap_metrics, pearson

IDENTITY_COLUMNS: tuple[str, ...] = ("sequence_id", "sample_token", "track_id")
REQUIRED_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
    "prediction_expansion",
    "zero_events_expansion",
    "shuffled_mean_expansion",
)


@dataclass(frozen=True)
class FixedFusionConfig:
    alpha: float = 0.50
    alpha_sweep_points: int = 21
    max_abs_expansion: float = 0.25
    ttc_clip_seconds: float = 60.0
    minimum_abs_expansion: float = 1.0e-4
    per_sequence_negative_min_count: int = 20
    pearson_improvement_gate: float = 0.015
    mae_relative_improvement_gate: float = 0.03
    weighted_mid_relative_improvement_gate: float = 0.03
    balanced_sign_gate: float = 0.73
    negative_accuracy_gate: float = 0.60
    minimum_sequence_pearson_gate: float = 0.45
    minimum_sequence_negative_accuracy_gate: float = 0.20
    zero_event_pearson_drop_gate: float = 0.50
    shuffled_event_pearson_drop_gate: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0,1]")
        if self.alpha_sweep_points < 2:
            raise ValueError("alpha_sweep_points must be at least 2")
        if not 0.0 < self.max_abs_expansion < 1.0:
            raise ValueError("max_abs_expansion must lie in (0,1)")
        if min(
            self.ttc_clip_seconds,
            self.minimum_abs_expansion,
            self.per_sequence_negative_min_count,
        ) <= 0:
            raise ValueError("v4.9 positive controls must be positive")


def convex_fusion(base: np.ndarray, dense: np.ndarray, alpha: float) -> np.ndarray:
    """Return ``(1-alpha) * base + alpha * dense`` with shape checks."""

    base_array = np.asarray(base, dtype=np.float64).reshape(-1)
    dense_array = np.asarray(dense, dtype=np.float64).reshape(-1)
    if base_array.shape != dense_array.shape:
        raise ValueError(f"fusion shape mismatch: {base_array.shape} != {dense_array.shape}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    return (1.0 - float(alpha)) * base_array + float(alpha) * dense_array


def _require_columns(frame: pd.DataFrame, *, name: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate identities")


def align_prediction_frames(
    base: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    split_name: str,
    tolerance: float = 1.0e-7,
) -> pd.DataFrame:
    """Align two expert prediction tables without relying on row order."""

    _require_columns(base, name=f"{split_name} base")
    _require_columns(dense, name=f"{split_name} dense")
    left = base.loc[:, REQUIRED_COLUMNS].copy()
    right = dense.loc[:, REQUIRED_COLUMNS].copy()
    merged = left.merge(
        right,
        on=list(IDENTITY_COLUMNS),
        how="outer",
        validate="one_to_one",
        suffixes=("_base", "_dense"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"{split_name} identity mismatch: {counts}")
    for column in ("delta_t_s", "target_ttc_s", "target_expansion"):
        delta = np.abs(
            merged[f"{column}_base"].to_numpy(dtype=np.float64)
            - merged[f"{column}_dense"].to_numpy(dtype=np.float64)
        )
        if np.any(delta > tolerance):
            raise ValueError(f"{split_name} target mismatch in {column}: max={float(delta.max())}")
    output = merged.loc[:, IDENTITY_COLUMNS].copy()
    output["delta_t_s"] = merged["delta_t_s_base"].to_numpy(dtype=np.float64)
    output["target_ttc_s"] = merged["target_ttc_s_base"].to_numpy(dtype=np.float64)
    output["target_expansion"] = merged["target_expansion_base"].to_numpy(dtype=np.float64)
    for column in ("prediction_expansion", "zero_events_expansion", "shuffled_mean_expansion"):
        output[f"base_{column}"] = merged[f"{column}_base"].to_numpy(dtype=np.float64)
        output[f"dense_{column}"] = merged[f"{column}_dense"].to_numpy(dtype=np.float64)
    return output.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def _per_sequence(
    rows: pd.DataFrame,
    prediction_column: str,
    *,
    config: FixedFusionConfig,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id, frame in rows.groupby("sequence_id", sort=True):
        target = frame["target_expansion"].to_numpy(dtype=np.float64)
        prediction = frame[prediction_column].to_numpy(dtype=np.float64)
        metrics = branch_metrics(
            target,
            prediction,
            frame["delta_t_s"].to_numpy(dtype=np.float64),
            ttc_clip_seconds=config.ttc_clip_seconds,
            minimum_abs_expansion=config.minimum_abs_expansion,
        )
        records.append({"sequence_id": sequence_id, **metrics})
    return pd.DataFrame.from_records(records)


def evaluate_prediction(
    rows: pd.DataFrame,
    prediction_column: str,
    *,
    config: FixedFusionConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    target = rows["target_expansion"].to_numpy(dtype=np.float64)
    prediction = rows[prediction_column].to_numpy(dtype=np.float64)
    metrics: dict[str, object] = branch_metrics(
        target,
        prediction,
        rows["delta_t_s"].to_numpy(dtype=np.float64),
        ttc_clip_seconds=config.ttc_clip_seconds,
        minimum_abs_expansion=config.minimum_abs_expansion,
    )
    metrics["official_eap"] = official_eap_metrics(
        target,
        prediction,
        rows["delta_t_s"].to_numpy(dtype=np.float64),
        rows["target_ttc_s"].to_numpy(dtype=np.float64),
        max_abs_expansion=config.max_abs_expansion,
    )
    per_sequence = _per_sequence(rows, prediction_column, config=config)
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


def add_fusion_columns(rows: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    result = rows.copy()
    for suffix in ("prediction_expansion", "zero_events_expansion", "shuffled_mean_expansion"):
        result[f"fused_{suffix}"] = convex_fusion(
            result[f"base_{suffix}"].to_numpy(dtype=np.float64),
            result[f"dense_{suffix}"].to_numpy(dtype=np.float64),
            alpha,
        )
    result["expert_disagreement"] = np.abs(
        result["base_prediction_expansion"] - result["dense_prediction_expansion"]
    )
    return result


def alpha_sweep(rows: pd.DataFrame, *, config: FixedFusionConfig) -> pd.DataFrame:
    """Diagnostic sweep.  Its optimum must never replace ``config.alpha``."""

    records: list[dict[str, object]] = []
    target = rows["target_expansion"].to_numpy(dtype=np.float64)
    delta_t = rows["delta_t_s"].to_numpy(dtype=np.float64)
    target_ttc = rows["target_ttc_s"].to_numpy(dtype=np.float64)
    base = rows["base_prediction_expansion"].to_numpy(dtype=np.float64)
    dense = rows["dense_prediction_expansion"].to_numpy(dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, config.alpha_sweep_points):
        prediction = convex_fusion(base, dense, float(alpha))
        branch = branch_metrics(
            target,
            prediction,
            delta_t,
            ttc_clip_seconds=config.ttc_clip_seconds,
            minimum_abs_expansion=config.minimum_abs_expansion,
        )
        official = official_eap_metrics(
            target,
            prediction,
            delta_t,
            target_ttc,
            max_abs_expansion=config.max_abs_expansion,
        )
        records.append(
            {
                "alpha": float(alpha),
                "pearson": branch["pearson"],
                "expansion_mae": branch["expansion_mae"],
                "balanced_sign_accuracy": branch["balanced_sign_accuracy"],
                "negative_accuracy": branch["negative_accuracy"],
                "weighted_mid": official["weighted_mid"],
            }
        )
    return pd.DataFrame.from_records(records)


def fusion_gates(
    *,
    base: Mapping[str, object],
    dense: Mapping[str, object],
    fused: Mapping[str, object],
    dependence: Mapping[str, float],
    config: FixedFusionConfig,
) -> dict[str, bool]:
    best_pearson = max(float(base["pearson"]), float(dense["pearson"]))
    best_mae = min(float(base["expansion_mae"]), float(dense["expansion_mae"]))
    base_mid = float(cast_mapping(base["official_eap"])["weighted_mid"])
    dense_mid = float(cast_mapping(dense["official_eap"])["weighted_mid"])
    best_mid = min(base_mid, dense_mid)
    fused_mid = float(cast_mapping(fused["official_eap"])["weighted_mid"])
    return {
        "pearson_improvement": float(fused["pearson"]) - best_pearson
        >= config.pearson_improvement_gate,
        "mae_relative_improvement": (best_mae - float(fused["expansion_mae"]))
        / max(best_mae, 1.0e-12)
        >= config.mae_relative_improvement_gate,
        "weighted_mid_relative_improvement": (best_mid - fused_mid) / max(best_mid, 1.0e-12)
        >= config.weighted_mid_relative_improvement_gate,
        "balanced_sign": float(fused["balanced_sign_accuracy"]) >= config.balanced_sign_gate,
        "negative_accuracy": float(fused["negative_accuracy"]) >= config.negative_accuracy_gate,
        "minimum_sequence_pearson": float(fused["minimum_sequence_pearson"])
        >= config.minimum_sequence_pearson_gate,
        "minimum_sequence_negative_accuracy": float(
            fused["minimum_sequence_negative_accuracy"]
        )
        >= config.minimum_sequence_negative_accuracy_gate,
        "zero_event_dependence": float(dependence["zero_event_pearson_drop"])
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": float(dependence["shuffled_event_pearson_drop"])
        >= config.shuffled_event_pearson_drop_gate,
    }


def cast_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value


def dependence_metrics(rows: pd.DataFrame) -> dict[str, float]:
    target = rows["target_expansion"].to_numpy(dtype=np.float64)
    prediction = rows["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    zero = rows["fused_zero_events_expansion"].to_numpy(dtype=np.float64)
    shuffled = rows["fused_shuffled_mean_expansion"].to_numpy(dtype=np.float64)
    prediction_pearson = pearson(target, prediction)
    return {
        "zero_event_pearson_drop": prediction_pearson - pearson(target, zero),
        "zero_event_mean_abs_change": float(np.mean(np.abs(prediction - zero))),
        "shuffled_event_pearson_drop": prediction_pearson - pearson(target, shuffled),
        "shuffled_event_mean_abs_change": float(np.mean(np.abs(prediction - shuffled))),
    }


__all__ = [
    "FixedFusionConfig",
    "add_fusion_columns",
    "align_prediction_frames",
    "alpha_sweep",
    "convex_fusion",
    "dependence_metrics",
    "evaluate_prediction",
    "fusion_gates",
]
