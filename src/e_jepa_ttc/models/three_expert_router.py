"""Frozen three-expert A5/C2F/PAIR router for Stage 61."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.evaluation.scientific_recovery_v8 import raw_mid_per_sample

BASE8_FEATURES: tuple[str, ...] = (
    "shared_event_count_log1p",
    "shared_event_rate_log1p",
    "a5_flow",
    "a5_margin",
    "a5_log_variance",
    "c2f_flow",
    "c2f_margin",
    "c2f_log_variance",
)
PHASE9_FEATURES: tuple[str, ...] = (
    "a5_benchmark_phase",
    "c2f_benchmark_phase",
    "pair_benchmark_phase",
    "pair_minus_a5_phase",
    "pair_minus_c2f_phase",
    "c2f_minus_a5_phase",
    "abs_pair_minus_a5_phase",
    "abs_pair_minus_c2f_phase",
    "abs_c2f_minus_a5_phase",
)
PHASE17_FEATURES = BASE8_FEATURES + PHASE9_FEATURES
EXPERTS: tuple[str, ...] = ("A5", "C2F", "PAIR")


class ThreeExpertRouterError(ValueError):
    """Raised when the Stage 61 router contract is violated."""


@dataclass(frozen=True)
class ThreeExpertRouterFit:
    """Fitted pipeline and signed, serializable fit signature."""

    pipeline: Pipeline
    signature: dict[str, Any]


def validate_feature_frame(frame: pd.DataFrame, *, phase_features: bool) -> pd.DataFrame:
    """Accept exactly BASE8 or PHASE17 in the frozen order."""

    expected = PHASE17_FEATURES if phase_features else BASE8_FEATURES
    if tuple(frame.columns.astype(str)) != expected:
        raise ThreeExpertRouterError(f"router feature order mismatch: expected={list(expected)}")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ThreeExpertRouterError("router features must be a non-empty finite matrix")
    return pd.DataFrame(values, columns=expected, index=frame.index)


def three_expert_labels_and_weights(
    *, target: np.ndarray, predictions: np.ndarray, base_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-tie argmin labels and regret-weighted fit mass."""

    target = np.asarray(target, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    base_weights = np.asarray(base_weights, dtype=np.float64)
    if predictions.shape != (target.size, 3) or base_weights.shape != target.shape:
        raise ThreeExpertRouterError("target/prediction/weight shapes disagree")
    losses = np.stack(
        [raw_mid_per_sample(target, predictions[:, index]) for index in range(3)], axis=1
    )
    if not np.isfinite(losses).all() or not np.isfinite(base_weights).all():
        raise ThreeExpertRouterError("router labels require finite losses and weights")
    labels = np.argmin(losses, axis=1).astype(np.int64)  # np.argmin gives A5→C2F→PAIR ties.
    ordered = np.sort(losses, axis=1)
    weights = base_weights * (ordered[:, 1] - ordered[:, 0])
    for label in range(3):
        if not np.any(labels == label) or float(weights[labels == label].sum()) <= 0.0:
            raise ThreeExpertRouterError(f"expert class {EXPERTS[label]} has no positive fit mass")
    return labels, weights


def fit_three_expert_router(
    features: pd.DataFrame,
    *,
    phase_features: bool,
    target: np.ndarray,
    predictions: np.ndarray,
    base_weights: np.ndarray,
    sample_tokens: tuple[str, ...],
    seed: int,
) -> ThreeExpertRouterFit:
    """Fit the sole frozen multinomial softmax router."""

    checked = validate_feature_frame(features, phase_features=phase_features)
    if len(sample_tokens) != len(checked) or len(set(sample_tokens)) != len(sample_tokens):
        raise ThreeExpertRouterError("fit tokens must be unique and row-aligned")
    labels, weights = three_expert_labels_and_weights(
        target=target, predictions=predictions, base_weights=base_weights
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "router",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    pipeline.fit(checked, labels, router__sample_weight=weights)
    classifier = pipeline.named_steps["router"]
    scaler = pipeline.named_steps["scale"]
    signature = sign_artifact(
        {
            "artifact_type": "stage61_three_expert_router_signature_v1",
            "seed": int(seed),
            "experts": list(EXPERTS),
            "feature_order": list(checked.columns),
            "fit_rows": len(checked),
            "fit_tokens_sha256": hash_sorted_token_strings(sample_tokens),
            "fit_label_counts": {
                name: int(np.count_nonzero(labels == index)) for index, name in enumerate(EXPERTS)
            },
            "fit_weight_sum": float(weights.sum()),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "coef": classifier.coef_.astype(float).tolist(),
            "intercept": classifier.intercept_.astype(float).tolist(),
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "hard_argmax": True,
            "inner_oof_only": True,
        }
    )
    return ThreeExpertRouterFit(pipeline=pipeline, signature=signature)


def predict_expert_index(fit: ThreeExpertRouterFit, features: pd.DataFrame) -> np.ndarray:
    """Return hard expert indices after strict schema validation."""

    phase = tuple(features.columns.astype(str)) == PHASE17_FEATURES
    checked = validate_feature_frame(features, phase_features=phase)
    result = fit.pipeline.predict(checked).astype(np.int64)
    if not np.all(np.isin(result, (0, 1, 2))):
        raise ThreeExpertRouterError("router emitted an unknown expert class")
    return result


__all__ = [
    "BASE8_FEATURES",
    "EXPERTS",
    "PHASE17_FEATURES",
    "PHASE9_FEATURES",
    "ThreeExpertRouterError",
    "ThreeExpertRouterFit",
    "fit_three_expert_router",
    "predict_expert_index",
    "three_expert_labels_and_weights",
    "validate_feature_frame",
]
