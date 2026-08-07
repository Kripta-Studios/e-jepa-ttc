"""Locked multiseed replication utilities for the v4.13 dual-head correction.

V4.14 does not retune the v4.13 fusion.  It trains the exact odd directional
probe with true seeds 7/13/23, applies the already locked v4.13 rule to every
seed, and forms a conservative consensus from the median receding probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from e_jepa_ttc.object_event_v4_13 import (
    ObjectEventV413Config,
    conservative_dual_head_prediction,
)

IDENTITY_COLUMNS: tuple[str, ...] = ("sequence_id", "sample_token", "track_id")
TARGET_COLUMNS: tuple[str, ...] = ("delta_t_s", "target_ttc_s", "target_expansion")


@dataclass(frozen=True)
class V414AggregateConfig:
    seeds: tuple[int, ...] = (7, 13, 23)
    minimum_seed_pass_count: int = 2
    mean_seed_pearson_gate: float = 0.66
    worst_seed_pearson_gate: float = 0.64
    mean_seed_negative_accuracy_gate: float = 0.66
    worst_seed_negative_accuracy_gate: float = 0.60
    mean_seed_min_sequence_negative_accuracy_gate: float = 0.30
    worst_seed_min_sequence_negative_accuracy_gate: float = 0.15
    consensus_pearson_gate: float = 0.67
    consensus_track_bootstrap_lower_gate: float = 0.58
    consensus_weighted_mid_gate: float = 200.0
    consensus_weighted_rte_relative_ceiling: float = 1.5
    consensus_saturation_max_increase: float = 0.06
    consensus_expansion_mae_gate: float = 0.0157
    consensus_balanced_sign_gate: float = 0.765
    consensus_negative_accuracy_gate: float = 0.68
    consensus_min_sequence_pearson_gate: float = 0.57
    consensus_min_sequence_negative_accuracy_gate: float = 0.35
    consensus_positive_accuracy_gate: float = 0.83
    consensus_override_rate_gate: float = 0.08
    pairwise_selective_prediction_pearson_gate: float = 0.90
    pairwise_negative_probability_pearson_gate: float = 0.45
    mean_sample_prediction_std_gate: float = 0.008
    zero_event_pearson_drop_gate: float = 0.55
    shuffled_event_pearson_drop_gate: float = 0.55
    per_sequence_negative_min_count: int = 20
    track_bootstrap_repeats: int = 3000

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.14 requires at least three unique seeds")
        if not 1 <= self.minimum_seed_pass_count <= len(self.seeds):
            raise ValueError("minimum_seed_pass_count is outside the seed count")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")
        if self.track_bootstrap_repeats < 100:
            raise ValueError("track_bootstrap_repeats must be at least 100")


def _normalize(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        "baseline_prediction_expansion",
        "negative_probability",
        "selective_prediction_expansion",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate identities")
    return frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def align_selective_seeds(
    seed_frames: Mapping[int, pd.DataFrame],
    *,
    fusion_config: ObjectEventV413Config,
) -> pd.DataFrame:
    """Align seed predictions and apply a pre-registered median-probability consensus.

    Because the locked v4.13 magnitude baseline is identical across seeds, the
    median probability requires at least two of three directional probes to be
    extremely confident before the positive-to-negative override fires.
    """
    if len(seed_frames) < 2:
        raise ValueError("at least two seed frames are required")
    seeds = sorted(seed_frames)
    normalized = {seed: _normalize(frame, name=f"seed {seed}") for seed, frame in seed_frames.items()}
    first = normalized[seeds[0]]
    result = first.loc[:, [*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
    baseline = first["baseline_prediction_expansion"].to_numpy(dtype=np.float64)
    result["baseline_prediction_expansion"] = baseline

    for seed in seeds:
        frame = normalized[seed]
        if not result.loc[:, list(IDENTITY_COLUMNS)].equals(frame.loc[:, list(IDENTITY_COLUMNS)]):
            raise ValueError(f"identity mismatch for seed {seed}")
        for column in TARGET_COLUMNS:
            if not np.allclose(
                result[column].to_numpy(dtype=np.float64),
                frame[column].to_numpy(dtype=np.float64),
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError(f"target mismatch in {column} for seed {seed}")
        seed_baseline = frame["baseline_prediction_expansion"].to_numpy(dtype=np.float64)
        if not np.allclose(seed_baseline, baseline, atol=1.0e-8, rtol=0.0):
            raise ValueError(f"locked magnitude baseline differs for seed {seed}")
        result[f"negative_probability_seed_{seed}"] = frame["negative_probability"].to_numpy(
            dtype=np.float64
        )
        result[f"selective_prediction_seed_{seed}"] = frame[
            "selective_prediction_expansion"
        ].to_numpy(dtype=np.float64)

    probability_columns = [f"negative_probability_seed_{seed}" for seed in seeds]
    prediction_columns = [f"selective_prediction_seed_{seed}" for seed in seeds]
    result["consensus_negative_probability"] = result[probability_columns].median(axis=1)
    prediction, directional, blend, override = conservative_dual_head_prediction(
        baseline,
        result["consensus_negative_probability"].to_numpy(dtype=np.float64),
        config=fusion_config,
    )
    result["consensus_directional_expansion"] = directional
    result["consensus_blend"] = blend
    result["consensus_override"] = override
    result["consensus_prediction_expansion"] = prediction
    result["seed_selective_prediction_std"] = result[prediction_columns].std(axis=1, ddof=0)
    result["seed_negative_probability_std"] = result[probability_columns].std(axis=1, ddof=0)
    return result


def pairwise_seed_metrics(seed_frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    seeds = sorted(seed_frames)
    normalized = {seed: _normalize(frame, name=f"seed {seed}") for seed, frame in seed_frames.items()}
    rows: list[dict[str, float | int]] = []
    for index, seed_a in enumerate(seeds):
        a = normalized[seed_a]
        for seed_b in seeds[index + 1 :]:
            b = normalized[seed_b]
            if not a.loc[:, list(IDENTITY_COLUMNS)].equals(b.loc[:, list(IDENTITY_COLUMNS)]):
                raise ValueError(f"identity mismatch between seeds {seed_a} and {seed_b}")
            pa = a["selective_prediction_expansion"].to_numpy(dtype=np.float64)
            pb = b["selective_prediction_expansion"].to_numpy(dtype=np.float64)
            qa = a["negative_probability"].to_numpy(dtype=np.float64)
            qb = b["negative_probability"].to_numpy(dtype=np.float64)
            rows.append({
                "seed_a": seed_a,
                "seed_b": seed_b,
                "selective_prediction_pearson": _pearson(pa, pb),
                "negative_probability_pearson": _pearson(qa, qb),
                "mean_abs_prediction_difference": float(np.mean(np.abs(pa - pb))),
                "override_agreement": float(np.mean(
                    (qa >= 0.985) == (qb >= 0.985)
                )),
            })
    return pd.DataFrame(rows)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) < 2 or float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def robustness_gates(
    *,
    per_seed: pd.DataFrame,
    consensus: Mapping[str, float],
    baseline: Mapping[str, float],
    diagnostics: Mapping[str, float],
    bootstrap: Mapping[str, float],
    pairwise: pd.DataFrame,
    config: V414AggregateConfig,
) -> dict[str, bool]:
    return {
        "minimum_seed_pass_count": int(per_seed["passed"].sum()) >= config.minimum_seed_pass_count,
        "mean_seed_pearson": float(per_seed["pearson"].mean()) >= config.mean_seed_pearson_gate,
        "worst_seed_pearson": float(per_seed["pearson"].min()) >= config.worst_seed_pearson_gate,
        "mean_seed_negative_accuracy": float(per_seed["negative_accuracy"].mean())
        >= config.mean_seed_negative_accuracy_gate,
        "worst_seed_negative_accuracy": float(per_seed["negative_accuracy"].min())
        >= config.worst_seed_negative_accuracy_gate,
        "mean_seed_min_sequence_negative_accuracy": float(
            per_seed["minimum_sequence_negative_accuracy"].mean()
        ) >= config.mean_seed_min_sequence_negative_accuracy_gate,
        "worst_seed_min_sequence_negative_accuracy": float(
            per_seed["minimum_sequence_negative_accuracy"].min()
        ) >= config.worst_seed_min_sequence_negative_accuracy_gate,
        "consensus_pearson": float(consensus["pearson"]) >= config.consensus_pearson_gate,
        "consensus_track_bootstrap_lower": float(bootstrap["lower_95"])
        >= config.consensus_track_bootstrap_lower_gate,
        "consensus_weighted_mid": float(consensus["weighted_mid"])
        <= config.consensus_weighted_mid_gate,
        "consensus_weighted_rte_relative": float(consensus["weighted_rte_percent"])
        <= float(baseline["weighted_rte_percent"])
        * config.consensus_weighted_rte_relative_ceiling,
        "consensus_saturation": float(consensus["ttc_saturation_rate"])
        <= float(baseline["ttc_saturation_rate"])
        + config.consensus_saturation_max_increase,
        "consensus_expansion_mae": float(consensus["expansion_mae"])
        <= config.consensus_expansion_mae_gate,
        "consensus_balanced_sign": float(consensus["balanced_sign_accuracy"])
        >= config.consensus_balanced_sign_gate,
        "consensus_negative_accuracy": float(consensus["negative_accuracy"])
        >= config.consensus_negative_accuracy_gate,
        "consensus_min_sequence_pearson": float(consensus["minimum_sequence_pearson"])
        >= config.consensus_min_sequence_pearson_gate,
        "consensus_min_sequence_negative_accuracy": float(
            consensus["minimum_sequence_negative_accuracy"]
        ) >= config.consensus_min_sequence_negative_accuracy_gate,
        "consensus_positive_accuracy": float(consensus["positive_accuracy"])
        >= config.consensus_positive_accuracy_gate,
        "consensus_override_rate": float(diagnostics["override_rate"])
        <= config.consensus_override_rate_gate,
        "pairwise_selective_prediction_pearson": float(
            pairwise["selective_prediction_pearson"].min()
        ) >= config.pairwise_selective_prediction_pearson_gate,
        "pairwise_negative_probability_pearson": float(
            pairwise["negative_probability_pearson"].min()
        ) >= config.pairwise_negative_probability_pearson_gate,
        "mean_sample_prediction_std": float(diagnostics["mean_sample_prediction_std"])
        <= config.mean_sample_prediction_std_gate,
        "zero_event_dependence": float(diagnostics["zero_event_pearson_drop"])
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": float(diagnostics["shuffled_event_pearson_drop"])
        >= config.shuffled_event_pearson_drop_gate,
    }


__all__ = [
    "IDENTITY_COLUMNS",
    "TARGET_COLUMNS",
    "V414AggregateConfig",
    "align_selective_seeds",
    "pairwise_seed_metrics",
    "robustness_gates",
]
