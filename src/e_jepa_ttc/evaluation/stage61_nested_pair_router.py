"""Leakage-closed Stage 61 router assembly and evaluation primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics
from e_jepa_ttc.evaluation.scientific_recovery_v8 import raw_mid_per_sample
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase
from e_jepa_ttc.models.three_expert_router import (
    BASE8_FEATURES,
    PHASE17_FEATURES,
    ThreeExpertRouterFit,
    predict_expert_index,
)


def phase_from_ttc(values: np.ndarray) -> np.ndarray:
    """Convert physical TTC to benchmark phase and fail outside its real domain."""

    phase, valid = ttc_to_benchmark_phase(
        torch.as_tensor(values, dtype=torch.float64), metric_delta_t_s=0.1
    )
    if not bool(valid.all()):
        raise ValueError("TTC lies outside the signed benchmark-phase domain")
    return phase.numpy()


def build_router_features(
    a5: pd.DataFrame, c2f: pd.DataFrame, pair_prediction_ttc: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact BASE8 and PHASE17 frames from row-aligned inference fields."""

    if a5["token_id"].astype(str).tolist() != c2f["token_id"].astype(str).tolist():
        raise ValueError("A5/C2F tokens are not row-aligned")
    base = pd.DataFrame(
        {
            name: np.asarray(
                a5[name] if name.startswith(("shared_", "a5_")) else c2f[name],
                dtype=np.float64,
            )
            for name in BASE8_FEATURES
        }
    )
    a5_phase = phase_from_ttc(a5["prediction_ttc"].to_numpy(dtype=np.float64))
    c2f_phase = phase_from_ttc(c2f["prediction_ttc"].to_numpy(dtype=np.float64))
    pair_phase = phase_from_ttc(np.asarray(pair_prediction_ttc, dtype=np.float64))
    phase = pd.DataFrame(
        {
            "a5_benchmark_phase": a5_phase,
            "c2f_benchmark_phase": c2f_phase,
            "pair_benchmark_phase": pair_phase,
            "pair_minus_a5_phase": pair_phase - a5_phase,
            "pair_minus_c2f_phase": pair_phase - c2f_phase,
            "c2f_minus_a5_phase": c2f_phase - a5_phase,
            "abs_pair_minus_a5_phase": np.abs(pair_phase - a5_phase),
            "abs_pair_minus_c2f_phase": np.abs(pair_phase - c2f_phase),
            "abs_c2f_minus_a5_phase": np.abs(c2f_phase - a5_phase),
        }
    )
    combined = pd.concat((base, phase), axis=1)
    if tuple(combined.columns) != PHASE17_FEATURES:
        raise AssertionError("Stage 61 feature assembly order drifted")
    return base, combined


def select_predictions(
    fit: ThreeExpertRouterFit,
    features: pd.DataFrame,
    expert_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply hard router selection to exactly three row-aligned experts."""

    values = np.asarray(expert_predictions, dtype=np.float64)
    if values.shape != (len(features), 3) or not np.isfinite(values).all():
        raise ValueError("three-expert prediction matrix is invalid")
    selected = predict_expert_index(fit, features)
    return values[np.arange(len(values)), selected], selected


def prediction_frame(metadata: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Emit the canonical row-level evidence needed by metrics/bootstrap."""

    result = pd.DataFrame(
        {
            "sample_token": metadata["token_id"].astype(str),
            "sequence_id": metadata["sequence_id"].astype(str),
            "track_id": metadata["track_id"].astype(str),
            "target_ttc_s": metadata["target_ttc"].to_numpy(dtype=np.float64),
            "prediction_ttc_s": np.asarray(prediction, dtype=np.float64),
        }
    )
    result["scientific_mid_per_row"] = raw_mid_per_sample(
        result["target_ttc_s"].to_numpy(), result["prediction_ttc_s"].to_numpy()
    )
    result["finite"] = np.isfinite(result["prediction_ttc_s"])
    result["failure"] = (~result["finite"]) | (result["prediction_ttc_s"].abs() < 0.1)
    return result


def summarize_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute signed sequence-macro MiD and non-negotiable integrity rates."""

    metrics = sequence_macro_signed_metrics(
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        frame["prediction_ttc_s"].to_numpy(dtype=np.float64),
        frame["sequence_id"].astype(str),
    )
    return {
        **metrics,
        "rows": len(frame),
        "sequences": int(frame["sequence_id"].nunique()),
        "finite_fraction": float(frame["finite"].mean()),
        "failure_rate": float(frame["failure"].mean()),
    }


__all__ = [
    "build_router_features",
    "phase_from_ttc",
    "prediction_frame",
    "select_predictions",
    "summarize_prediction_frame",
]
