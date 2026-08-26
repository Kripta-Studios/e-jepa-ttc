"""Prospective, label-free-at-inference router between the A5 and C2F experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from e_jepa_ttc.artifacts.hashing import compute_artifact_hash, sign_artifact
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings

ROUTER_FEATURES: tuple[str, ...] = (
    "shared_event_count_log1p",
    "shared_event_rate_log1p",
    "a5_flow",
    "a5_margin",
    "a5_log_variance",
    "c2f_flow",
    "c2f_margin",
    "c2f_log_variance",
)
"""The frozen V8 router feature order. No tuning or extension is allowed."""

ROUTER_THRESHOLD = 0.5


class RouterSchemaError(ValueError):
    """Raised when input could leak labels or has a non-frozen schema."""


class RouterFitError(ValueError):
    """Raised when the frozen router cannot be fit validly."""


def _canonical_token_hash(tokens: tuple[str, ...]) -> str:
    return hash_sorted_token_strings(tokens)


def _canonical_weight_hash(weights: np.ndarray) -> str:
    """Bind the ordered effective fit weights to the signed training record."""

    return compute_artifact_hash({"effective_sample_weights": weights.tolist()})


def validate_router_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Require exactly the eight frozen, label-free router features in their order.

    Identity columns, targets, raw TTC predictions, folds and all other metadata must
    remain in a separate audit table.  Accepting a wider table here would make it too
    easy to accidentally pass a forbidden feature through the sklearn pipeline.
    """

    actual = tuple(str(column) for column in features.columns)
    if actual != ROUTER_FEATURES:
        missing = [name for name in ROUTER_FEATURES if name not in actual]
        unexpected = [name for name in actual if name not in ROUTER_FEATURES]
        raise RouterSchemaError(
            "Router features must be exactly the frozen V8 order; "
            f"missing={missing}, unexpected={unexpected}, actual={list(actual)}. "
            "Keep targets, raw predictions, identities, folds and future-derived values "
            "out of this frame."
        )
    values = features.to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != len(ROUTER_FEATURES):
        raise RouterSchemaError("Router feature matrix has an invalid shape.")
    if values.shape[0] == 0:
        raise RouterSchemaError("Router feature matrix is empty.")
    if not np.all(np.isfinite(values)):
        raise RouterSchemaError("Router features must be finite; no imputation is allowed.")
    return pd.DataFrame(values, columns=ROUTER_FEATURES, index=features.index)


def build_router_pipeline(seed: int) -> Pipeline:
    """Return the single preregistered sklearn pipeline used by V8."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "router",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class RouterSignature:
    """Signed parameters proving the fitted router and its training population."""

    payload: dict[str, Any]

    @property
    def artifact_sha256(self) -> str:
        """Return the signature hash produced from canonical JSON."""

        return str(self.payload["artifact_sha256"])


class CausalExpertRouter:
    """Hard-route A5/C2F predictions using only frozen label-free diagnostics."""

    def __init__(self, *, seed: int, threshold: float = ROUTER_THRESHOLD) -> None:
        if threshold != ROUTER_THRESHOLD:
            raise ValueError("The V8 router threshold is frozen at 0.5.")
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.pipeline = build_router_pipeline(self.seed)
        self._signature: RouterSignature | None = None

    @property
    def signature(self) -> RouterSignature:
        """Return the signed fit record after a successful fit."""

        if self._signature is None:
            raise RouterFitError("Router has not been fitted.")
        return self._signature

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray | pd.Series,
        *,
        sample_tokens: tuple[str, ...],
        effective_sample_weights: np.ndarray | pd.Series,
    ) -> CausalExpertRouter:
        """Fit only on inner-OOF rows with official macro-MiD effective weights."""

        checked = validate_router_feature_frame(features)
        y = np.asarray(labels, dtype=np.int64)
        if y.ndim != 1 or y.shape[0] != len(checked):
            raise RouterFitError("Router labels must be one-dimensional and row-aligned.")
        if len(sample_tokens) != len(checked) or len(set(sample_tokens)) != len(sample_tokens):
            raise RouterFitError("sample_tokens must be unique and match the OOF feature rows.")
        if not np.all(np.isin(y, (0, 1))):
            raise RouterFitError("Router labels must be binary values {0, 1}.")
        weights = np.asarray(effective_sample_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != len(checked):
            raise RouterFitError(
                "Effective router sample weights must be one-dimensional and row-aligned."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise RouterFitError("Effective router sample weights must be finite and non-negative.")
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise RouterFitError("Effective router sample weights must have positive total mass.")
        classes = np.unique(y)
        if classes.size != 2:
            raise RouterFitError(
                "Router inner OOF labels are degenerate; both A5-win (0) and C2F-win (1) "
                "rows are required before fitting LogisticRegression."
            )
        for outcome, label in ((0, "A5-win"), (1, "C2F-win")):
            class_weight = float(np.sum(weights[y == outcome]))
            if not np.isfinite(class_weight) or class_weight <= 0.0:
                raise RouterFitError(
                    f"Effective router sample weights require positive mass for {label} rows."
                )
        # Metadata routing is disabled by convention: pass fit metadata only to the
        # classifier so StandardScaler remains an unweighted feature transform.
        self.pipeline.fit(checked, y, router__sample_weight=weights)
        scaler = self.pipeline.named_steps["scale"]
        classifier = self.pipeline.named_steps["router"]
        payload: dict[str, Any] = {
            "artifact_type": "scientific_recovery_v8_causal_expert_router_signature_v2",
            "seed": self.seed,
            "threshold": self.threshold,
            "feature_order": list(ROUTER_FEATURES),
            "fit_rows": int(len(checked)),
            "fit_sample_tokens_sha256": _canonical_token_hash(sample_tokens),
            "fit_effective_sample_weights_sha256": _canonical_weight_hash(weights),
            "fit_effective_sample_weight_sum": total_weight,
            "fit_effective_sample_weight_min": float(np.min(weights)),
            "fit_effective_sample_weight_max": float(np.max(weights)),
            "fit_label_counts": {
                "a5": int(np.count_nonzero(y == 0)),
                "c2f": int(np.count_nonzero(y == 1)),
            },
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "coef": [[float(value) for value in row] for row in classifier.coef_],
            "intercept": [float(value) for value in classifier.intercept_],
            "sklearn": {
                "penalty": "l2",
                "C": 1.0,
                "class_weight": None,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": self.seed,
            },
            "fit_semantics": {
                "label": "1 iff C2F official raw MiD loss is strictly lower than A5",
                "effective_sample_weight": (
                    "official_macro_mid_row_weight * abs(raw_mid_loss_c2f - raw_mid_loss_a5)"
                ),
                "metadata_routing": (
                    "disabled; Pipeline.fit receives router__sample_weight only, "
                    "so StandardScaler is unweighted"
                ),
                "objective": "weighted binary logistic regression aligned to sequence-macro MiD",
            },
        }
        self._signature = RouterSignature(sign_artifact(payload))
        return self

    def predict_c2f_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(C2F is better) from the fitted, frozen pipeline."""

        if self._signature is None:
            raise RouterFitError("Router has not been fitted.")
        checked = validate_router_feature_frame(features)
        probabilities = self.pipeline.predict_proba(checked)
        class_positions = {int(label): index for index, label in enumerate(self.pipeline.classes_)}
        if set(class_positions) != {0, 1}:
            raise RouterFitError("Fitted router does not contain both frozen outcome classes.")
        return probabilities[:, class_positions[1]].astype(np.float64, copy=False)

    def choose_c2f(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the fixed 0.5 threshold for deterministic hard routing."""

        return self.predict_c2f_probability(features) >= self.threshold

    def route(
        self,
        features: pd.DataFrame,
        *,
        a5_prediction_ttc_s: np.ndarray,
        c2f_prediction_ttc_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return hard-routed TTC, C2F choice and C2F probability."""

        a5 = np.asarray(a5_prediction_ttc_s, dtype=np.float64)
        c2f = np.asarray(c2f_prediction_ttc_s, dtype=np.float64)
        if a5.ndim != 1 or c2f.ndim != 1 or a5.shape != c2f.shape:
            raise ValueError("A5 and C2F predictions must be matching one-dimensional arrays.")
        if len(features) != len(a5):
            raise ValueError("Prediction arrays must be row-aligned with router features.")
        if not np.all(np.isfinite(a5)) or not np.all(np.isfinite(c2f)):
            raise ValueError("Hard routing requires finite expert TTC predictions.")
        probability = self.predict_c2f_probability(features)
        choose_c2f = probability >= self.threshold
        return np.where(choose_c2f, c2f, a5), choose_c2f, probability


__all__ = [
    "ROUTER_FEATURES",
    "ROUTER_THRESHOLD",
    "CausalExpertRouter",
    "RouterFitError",
    "RouterSchemaError",
    "RouterSignature",
    "build_router_pipeline",
    "validate_router_feature_frame",
]
