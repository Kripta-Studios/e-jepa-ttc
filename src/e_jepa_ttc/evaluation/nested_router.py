"""Nested grouped cross-fitting primitives for the prospective V8 A5/C2F router."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.scientific_recovery_v8 import raw_mid_per_sample
from e_jepa_ttc.models.causal_expert_router import (
    ROUTER_FEATURES,
    CausalExpertRouter,
    RouterFitError,
)

IDENTITY_COLUMNS = ("sample_token", "sequence_id", "track_id")
LABEL_COLUMNS = ("target_ttc_s", "a5_prediction_ttc_s", "c2f_prediction_ttc_s")
INNER_OOF_COLUMNS = (
    *IDENTITY_COLUMNS,
    "inner_fold",
    *LABEL_COLUMNS,
    "sample_weight",
    *ROUTER_FEATURES,
)


@dataclass(frozen=True)
class InnerFold:
    """One grouped inner fold fixed before expert training begins."""

    index: int
    train_sequences: tuple[str, ...]
    dev_sequences: tuple[str, ...]


@dataclass(frozen=True)
class NestedRouterFit:
    """Fitted router plus the label-free outer inference contract."""

    router: CausalExpertRouter
    inner_folds: tuple[InnerFold, ...]
    inner_oof_tokens: tuple[str, ...]
    outer_dev_sequences: tuple[str, ...]


class NestedRouterIntegrityError(ValueError):
    """Raised when a nested split or OOF population violates the V8 protocol."""


def _unique_sequences(sequences: Iterable[str], *, label: str) -> tuple[str, ...]:
    ordered = tuple(sorted(str(item) for item in sequences))
    if not ordered:
        raise NestedRouterIntegrityError(f"{label} must not be empty.")
    if len(set(ordered)) != len(ordered):
        raise NestedRouterIntegrityError(f"{label} contains duplicate sequence IDs.")
    return ordered


def build_inner_folds(
    outer_folds: tuple[tuple[str, ...], ...], *, outer_fold_index: int
) -> tuple[InnerFold, ...]:
    """Build three lexicographically paired grouped inner folds for one outer fold."""

    if len(outer_folds) != 3:
        raise NestedRouterIntegrityError("V8 router requires exactly three outer folds.")
    normalized = tuple(
        _unique_sequences(fold, label=f"outer fold {index}")
        for index, fold in enumerate(outer_folds)
    )
    flattened = [sequence for fold in normalized for sequence in fold]
    if len(set(flattened)) != len(flattened):
        raise NestedRouterIntegrityError("Outer folds overlap by sequence ID.")
    if not 0 <= outer_fold_index < len(normalized):
        raise NestedRouterIntegrityError("outer_fold_index is out of range.")
    outer_dev = normalized[outer_fold_index]
    outer_train_groups = tuple(
        fold for index, fold in enumerate(normalized) if index != outer_fold_index
    )
    if len(outer_dev) != 3 or any(len(group) != 3 for group in outer_train_groups):
        raise NestedRouterIntegrityError(
            "V8 router requires three sequences per outer fold to form paired inner dev folds."
        )
    outer_train = tuple(sorted(sequence for group in outer_train_groups for sequence in group))
    inner_folds: list[InnerFold] = []
    for index in range(3):
        dev = tuple(sorted((outer_train_groups[0][index], outer_train_groups[1][index])))
        train = tuple(sequence for sequence in outer_train if sequence not in set(dev))
        inner_folds.append(InnerFold(index=index, train_sequences=train, dev_sequences=dev))
    validate_inner_folds(
        inner_folds, outer_train_sequences=outer_train, outer_dev_sequences=outer_dev
    )
    return tuple(inner_folds)


def validate_inner_folds(
    inner_folds: Iterable[InnerFold],
    *,
    outer_train_sequences: Iterable[str],
    outer_dev_sequences: Iterable[str],
) -> None:
    """Fail closed on overlap, missing coverage or accidental outer-dev use."""

    outer_train = set(_unique_sequences(outer_train_sequences, label="outer train"))
    outer_dev = set(_unique_sequences(outer_dev_sequences, label="outer dev"))
    if outer_train & outer_dev:
        raise NestedRouterIntegrityError("Outer train and outer dev overlap.")
    frozen = tuple(inner_folds)
    if len(frozen) != 3:
        raise NestedRouterIntegrityError("V8 router requires exactly three inner folds.")
    seen_dev: set[str] = set()
    for fold in frozen:
        train = set(fold.train_sequences)
        dev = set(fold.dev_sequences)
        if train & dev:
            raise NestedRouterIntegrityError(f"Inner fold {fold.index} train/dev overlap.")
        if not train <= outer_train or not dev <= outer_train:
            raise NestedRouterIntegrityError(
                f"Inner fold {fold.index} uses a sequence outside outer train."
            )
        if (train | dev) != outer_train:
            raise NestedRouterIntegrityError(
                f"Inner fold {fold.index} does not partition outer train."
            )
        if dev & outer_dev:
            raise NestedRouterIntegrityError(f"Inner fold {fold.index} illegally uses outer dev.")
        if seen_dev & dev:
            raise NestedRouterIntegrityError("Inner dev sequences are not disjoint.")
        seen_dev.update(dev)
    if seen_dev != outer_train:
        raise NestedRouterIntegrityError("Inner dev folds do not cover outer train exactly once.")


def router_labels_from_official_error(
    *, target_ttc_s: np.ndarray, a5_prediction_ttc_s: np.ndarray, c2f_prediction_ttc_s: np.ndarray
) -> np.ndarray:
    """Return one only where C2F has strictly lower official raw MiD error than A5."""

    target = np.asarray(target_ttc_s, dtype=np.float64)
    a5 = np.asarray(a5_prediction_ttc_s, dtype=np.float64)
    c2f = np.asarray(c2f_prediction_ttc_s, dtype=np.float64)
    if target.ndim != 1 or target.shape != a5.shape or target.shape != c2f.shape:
        raise NestedRouterIntegrityError("Target and expert predictions must be matching vectors.")
    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(a5))
        or not np.all(np.isfinite(c2f))
    ):
        raise NestedRouterIntegrityError(
            "Official router labels require finite targets and expert predictions."
        )
    a5_error = raw_mid_per_sample(target, a5)
    c2f_error = raw_mid_per_sample(target, c2f)
    if not np.all(np.isfinite(a5_error)) or not np.all(np.isfinite(c2f_error)):
        raise NestedRouterIntegrityError(
            "Official raw MiD was non-finite for router label construction."
        )
    return (c2f_error < a5_error).astype(np.int64)


def effective_router_fit_weights(
    *,
    official_macro_mid_row_weight: np.ndarray,
    raw_mid_loss_a5: np.ndarray,
    raw_mid_loss_c2f: np.ndarray,
) -> np.ndarray:
    """Return official macro-MiD mass scaled by the A5/C2F row-level regret.

    ``official_macro_mid_row_weight`` is frozen upstream as the official signed-bucket
    coefficient divided by nine sequences and then by the sequence-bucket row count.
    The returned effective fit weight therefore makes a wrong routing decision cost its
    exact sequence-macro MiD regret instead of one row-accuracy unit.
    """

    base = np.asarray(official_macro_mid_row_weight, dtype=np.float64)
    a5_loss = np.asarray(raw_mid_loss_a5, dtype=np.float64)
    c2f_loss = np.asarray(raw_mid_loss_c2f, dtype=np.float64)
    if base.ndim != 1 or base.shape != a5_loss.shape or base.shape != c2f_loss.shape:
        raise NestedRouterIntegrityError(
            "Official macro-MiD weights and raw expert losses must be matching vectors."
        )
    if not np.all(np.isfinite(base)) or np.any(base < 0.0):
        raise NestedRouterIntegrityError(
            "Official macro-MiD row weights must be finite and non-negative."
        )
    if (
        not np.all(np.isfinite(a5_loss))
        or not np.all(np.isfinite(c2f_loss))
        or np.any(a5_loss < 0.0)
        or np.any(c2f_loss < 0.0)
    ):
        raise NestedRouterIntegrityError(
            "Official raw MiD expert losses must be finite and non-negative."
        )
    effective = base * np.abs(c2f_loss - a5_loss)
    if not np.all(np.isfinite(effective)) or np.any(effective < 0.0):
        raise NestedRouterIntegrityError(
            "Effective router fit weights must be finite and non-negative."
        )
    return effective


def validate_inner_oof_frame(
    frame: pd.DataFrame,
    *,
    inner_folds: tuple[InnerFold, ...],
    outer_dev_sequences: Iterable[str],
) -> pd.DataFrame:
    """Validate identity, frozen schema and strict inner-OOF sequence ownership."""

    actual = tuple(str(column) for column in frame.columns)
    if actual != INNER_OOF_COLUMNS:
        raise NestedRouterIntegrityError(
            "Inner OOF router frame must use the exact frozen schema; "
            f"expected={list(INNER_OOF_COLUMNS)}, actual={list(actual)}."
        )
    checked = frame.copy()
    if checked.empty:
        raise NestedRouterIntegrityError("Inner OOF router frame is empty.")
    for column in IDENTITY_COLUMNS:
        if checked[column].isna().any() or checked[column].astype(str).str.len().eq(0).any():
            raise NestedRouterIntegrityError(
                f"Inner OOF identity column {column!r} has missing values."
            )
        checked[column] = checked[column].astype(str)
    if checked["sample_token"].duplicated().any():
        raise NestedRouterIntegrityError("Inner OOF sample_token values must be unique.")
    checked = checked.sort_values("sample_token", kind="stable").reset_index(drop=True)
    checked["inner_fold"] = pd.to_numeric(checked["inner_fold"], errors="raise").astype(np.int64)
    fold_by_index = {fold.index: fold for fold in inner_folds}
    if set(checked["inner_fold"].unique()) != set(fold_by_index):
        raise NestedRouterIntegrityError(
            "Inner OOF frame does not contain exactly the three frozen inner folds."
        )
    outer_dev = set(_unique_sequences(outer_dev_sequences, label="outer dev"))
    for index, fold in fold_by_index.items():
        sequences = set(checked.loc[checked["inner_fold"] == index, "sequence_id"])
        if not sequences:
            raise NestedRouterIntegrityError(f"Inner fold {index} has no OOF rows.")
        if sequences & outer_dev:
            raise NestedRouterIntegrityError("Outer dev rows cannot be used to fit the router.")
        if sequences != set(fold.dev_sequences):
            raise NestedRouterIntegrityError(
                f"Inner fold {index} does not contain exactly its frozen inner dev sequences."
            )
    for column in (*LABEL_COLUMNS, "sample_weight", *ROUTER_FEATURES):
        checked[column] = pd.to_numeric(checked[column], errors="raise").astype(np.float64)
        if not np.all(np.isfinite(checked[column].to_numpy())):
            raise NestedRouterIntegrityError(f"Inner OOF field {column!r} must be finite.")
    if np.any(checked["sample_weight"].to_numpy(dtype=np.float64) < 0.0):
        raise NestedRouterIntegrityError("Inner OOF sample_weight must be non-negative.")
    return checked


def fit_router_from_inner_oof(
    frame: pd.DataFrame,
    *,
    inner_folds: tuple[InnerFold, ...],
    outer_dev_sequences: Iterable[str],
    seed: int,
) -> NestedRouterFit:
    """Fit the router solely on validated inner OOF predictions for one outer fold."""

    checked = validate_inner_oof_frame(
        frame, inner_folds=inner_folds, outer_dev_sequences=outer_dev_sequences
    )
    labels = router_labels_from_official_error(
        target_ttc_s=checked["target_ttc_s"].to_numpy(),
        a5_prediction_ttc_s=checked["a5_prediction_ttc_s"].to_numpy(),
        c2f_prediction_ttc_s=checked["c2f_prediction_ttc_s"].to_numpy(),
    )
    effective_weights = effective_router_fit_weights(
        official_macro_mid_row_weight=checked["sample_weight"].to_numpy(),
        raw_mid_loss_a5=raw_mid_per_sample(
            checked["target_ttc_s"].to_numpy(),
            checked["a5_prediction_ttc_s"].to_numpy(),
        ),
        raw_mid_loss_c2f=raw_mid_per_sample(
            checked["target_ttc_s"].to_numpy(),
            checked["c2f_prediction_ttc_s"].to_numpy(),
        ),
    )
    router = CausalExpertRouter(seed=seed)
    try:
        router.fit(
            checked.loc[:, ROUTER_FEATURES],
            labels,
            sample_tokens=tuple(checked["sample_token"]),
            effective_sample_weights=effective_weights,
        )
    except RouterFitError as error:
        raise NestedRouterIntegrityError(
            f"Router fit rejected the inner OOF population: {error}"
        ) from error
    return NestedRouterFit(
        router=router,
        inner_folds=inner_folds,
        inner_oof_tokens=tuple(checked["sample_token"]),
        outer_dev_sequences=_unique_sequences(outer_dev_sequences, label="outer dev"),
    )


__all__ = [
    "IDENTITY_COLUMNS",
    "INNER_OOF_COLUMNS",
    "LABEL_COLUMNS",
    "InnerFold",
    "NestedRouterFit",
    "NestedRouterIntegrityError",
    "build_inner_folds",
    "effective_router_fit_weights",
    "fit_router_from_inner_oof",
    "router_labels_from_official_error",
    "validate_inner_folds",
    "validate_inner_oof_frame",
]
