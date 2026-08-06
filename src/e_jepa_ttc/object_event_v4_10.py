"""Multiseed robustness utilities for the fixed event-only v4.9 fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from e_jepa_ttc.object_event_v4_4 import pearson

IDENTITY_COLUMNS: tuple[str, ...] = ("sequence_id", "sample_token", "track_id")
TARGET_COLUMNS: tuple[str, ...] = ("delta_t_s", "target_ttc_s", "target_expansion")
FUSION_COLUMNS: tuple[str, ...] = (
    "fused_prediction_expansion",
    "fused_zero_events_expansion",
    "fused_shuffled_mean_expansion",
)


@dataclass(frozen=True)
class V410AggregateConfig:
    seeds: tuple[int, ...] = (7, 13, 23)
    require_all_seed_screens: bool = True
    mean_seed_pearson_gate: float = 0.60
    worst_seed_pearson_gate: float = 0.56
    seed_pearson_std_gate: float = 0.06
    mean_seed_balanced_sign_gate: float = 0.72
    worst_seed_balanced_sign_gate: float = 0.68
    mean_seed_negative_accuracy_gate: float = 0.58
    ensemble_pearson_gate: float = 0.64
    ensemble_track_bootstrap_lower_gate: float = 0.55
    ensemble_weighted_mid_gate: float = 205.0
    ensemble_balanced_sign_gate: float = 0.75
    ensemble_negative_accuracy_gate: float = 0.62
    ensemble_expansion_mae_gate: float = 0.0162
    ensemble_min_sequence_pearson_gate: float = 0.50
    ensemble_min_sequence_negative_accuracy_gate: float = 0.30
    pairwise_prediction_pearson_gate: float = 0.75
    mean_sample_prediction_std_gate: float = 0.018
    zero_event_pearson_drop_gate: float = 0.55
    shuffled_event_pearson_drop_gate: float = 0.55
    per_sequence_negative_min_count: int = 20
    track_bootstrap_repeats: int = 3000

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.10 requires at least three unique seeds")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")
        if self.track_bootstrap_repeats < 100:
            raise ValueError("track_bootstrap_repeats must be at least 100")


def require_fusion_columns(frame: pd.DataFrame, *, name: str) -> None:
    required = {*IDENTITY_COLUMNS, *TARGET_COLUMNS, *FUSION_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate identities")


def align_seed_fusions(seed_frames: dict[int, pd.DataFrame], *, split_name: str) -> pd.DataFrame:
    """Align fixed-fusion predictions by identity and average them across seeds."""

    if len(seed_frames) < 2:
        raise ValueError("at least two seed frames are required")
    seeds = sorted(seed_frames)
    normalized: dict[int, pd.DataFrame] = {}
    for seed, frame in seed_frames.items():
        require_fusion_columns(frame, name=f"{split_name} seed {seed}")
        normalized[seed] = frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
            drop=True
        )
    first = normalized[seeds[0]]
    result = first.loc[:, [*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
    for seed in seeds:
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
        result[f"prediction_seed_{seed}"] = frame["fused_prediction_expansion"].to_numpy(
            dtype=np.float64
        )
        result[f"zero_seed_{seed}"] = frame["fused_zero_events_expansion"].to_numpy(
            dtype=np.float64
        )
        result[f"shuffled_seed_{seed}"] = frame[
            "fused_shuffled_mean_expansion"
        ].to_numpy(dtype=np.float64)
    prediction_columns = [f"prediction_seed_{seed}" for seed in seeds]
    zero_columns = [f"zero_seed_{seed}" for seed in seeds]
    shuffled_columns = [f"shuffled_seed_{seed}" for seed in seeds]
    result["fused_prediction_expansion"] = result[prediction_columns].mean(axis=1)
    result["fused_zero_events_expansion"] = result[zero_columns].mean(axis=1)
    result["fused_shuffled_mean_expansion"] = result[shuffled_columns].mean(axis=1)
    result["seed_prediction_std"] = result[prediction_columns].std(axis=1, ddof=0)
    return result


def pairwise_seed_metrics(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    seeds = sorted(seed_frames)
    for index, seed_a in enumerate(seeds):
        prediction_a = seed_frames[seed_a]["fused_prediction_expansion"].to_numpy(
            dtype=np.float64
        )
        for seed_b in seeds[index + 1 :]:
            prediction_b = seed_frames[seed_b]["fused_prediction_expansion"].to_numpy(
                dtype=np.float64
            )
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "prediction_pearson": pearson(prediction_a, prediction_b),
                    "mean_abs_difference": float(np.mean(np.abs(prediction_a - prediction_b))),
                    "sign_agreement": float(
                        np.mean((prediction_a < 0.0) == (prediction_b < 0.0))
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def track_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    cluster_keys = frame["sequence_id"].astype(str) + "|" + frame["track_id"].astype(str)
    unique_keys = np.asarray(sorted(cluster_keys.unique()), dtype=object)
    indices = {
        str(key): np.flatnonzero(cluster_keys.to_numpy(dtype=object) == key)
        for key in unique_keys
    }
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_keys, size=len(unique_keys), replace=True)
        rows = np.concatenate([indices[str(key)] for key in sampled])
        values[repeat] = pearson(target[rows], prediction[rows])
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "cluster_count": int(len(unique_keys)),
        "repeats": int(repeats),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


__all__ = [
    "FUSION_COLUMNS",
    "IDENTITY_COLUMNS",
    "TARGET_COLUMNS",
    "V410AggregateConfig",
    "align_seed_fusions",
    "pairwise_seed_metrics",
    "require_fusion_columns",
    "track_cluster_bootstrap",
]
