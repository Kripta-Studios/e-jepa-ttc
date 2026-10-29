"""Unit contracts for the preregistered V8 prospective expert router."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import e_jepa_ttc.evaluation.nested_router as nested_router
from e_jepa_ttc.evaluation.nested_router import (
    INNER_OOF_COLUMNS,
    NestedRouterIntegrityError,
    bind_expert_oof_to_trainer_point_ttc,
    build_inner_folds,
    fit_router_from_inner_oof,
    integral_event_count_from_reconstructed,
    router_labels_from_official_error,
    routing_point_ttc,
    validate_inner_oof_frame,
)
from e_jepa_ttc.models.causal_expert_router import (
    ROUTER_FEATURES,
    CausalExpertRouter,
    RouterFitError,
    RouterSchemaError,
    validate_router_feature_frame,
)

OUTER_FOLDS = (
    ("sequence-00", "sequence-01", "sequence-02"),
    ("sequence-10", "sequence-11", "sequence-12"),
    ("sequence-20", "sequence-21", "sequence-22"),
)


def _features(rows: int = 12, *, offset: float = 0.0) -> pd.DataFrame:
    values = np.arange(rows * len(ROUTER_FEATURES), dtype=np.float64).reshape(rows, -1)
    return pd.DataFrame(values / 11.0 + offset, columns=ROUTER_FEATURES)


def _inner_oof_frame() -> tuple[pd.DataFrame, tuple[object, ...]]:
    inner_folds = build_inner_folds(OUTER_FOLDS, outer_fold_index=0)
    records: list[dict[str, object]] = []
    row = 0
    for fold in inner_folds:
        for sequence in fold.dev_sequences:
            for local_index in range(2):
                feature_values = _features(1, offset=float(row)).iloc[0].to_dict()
                c2f_better = row % 2 == 0
                records.append(
                    {
                        "sample_token": f"token-{row:03d}",
                        "sequence_id": sequence,
                        "track_id": f"track-{sequence}-{local_index}",
                        "inner_fold": fold.index,
                        "target_ttc_s": 2.0,
                        "a5_prediction_ttc_s": 2.8 if c2f_better else 2.05,
                        "c2f_prediction_ttc_s": 2.05 if c2f_better else 2.8,
                        "sample_weight": 1.0 / 18.0,
                        **feature_values,
                    }
                )
                row += 1
    return pd.DataFrame(records).loc[:, INNER_OOF_COLUMNS], inner_folds


def test_nested_folds_are_disjoint_and_lexicographically_paired() -> None:
    inner_folds = build_inner_folds(OUTER_FOLDS, outer_fold_index=1)

    assert [fold.dev_sequences for fold in inner_folds] == [
        ("sequence-00", "sequence-20"),
        ("sequence-01", "sequence-21"),
        ("sequence-02", "sequence-22"),
    ]
    assert set().union(*(set(fold.dev_sequences) for fold in inner_folds)) == {
        "sequence-00",
        "sequence-01",
        "sequence-02",
        "sequence-20",
        "sequence-21",
        "sequence-22",
    }
    assert all(not (set(fold.dev_sequences) & set(OUTER_FOLDS[1])) for fold in inner_folds)


def test_inner_oof_identity_and_outer_dev_leakage_are_rejected() -> None:
    frame, inner_folds = _inner_oof_frame()
    leaked = frame.copy()
    leaked.loc[0, "sequence_id"] = "sequence-00"

    with pytest.raises(NestedRouterIntegrityError, match="Outer dev rows"):
        validate_inner_oof_frame(
            leaked, inner_folds=inner_folds, outer_dev_sequences=OUTER_FOLDS[0]
        )

    duplicate = frame.copy()
    duplicate.loc[1, "sample_token"] = duplicate.loc[0, "sample_token"]
    with pytest.raises(NestedRouterIntegrityError, match="unique"):
        validate_inner_oof_frame(
            duplicate, inner_folds=inner_folds, outer_dev_sequences=OUTER_FOLDS[0]
        )


def test_router_feature_schema_rejects_forbidden_and_reordered_columns() -> None:
    features = _features()
    forbidden = features.assign(target_ttc_s=1.0)
    with pytest.raises(RouterSchemaError, match="exactly the frozen V8 order"):
        validate_router_feature_frame(forbidden)

    reordered = features.loc[:, tuple(reversed(ROUTER_FEATURES))]
    with pytest.raises(RouterSchemaError, match="exactly the frozen V8 order"):
        validate_router_feature_frame(reordered)


def test_router_label_uses_existing_official_raw_mid_formula() -> None:
    labels = router_labels_from_official_error(
        target_ttc_s=np.array([1.0, 2.0]),
        a5_prediction_ttc_s=np.array([1.1, 2.8]),
        c2f_prediction_ttc_s=np.array([1.3, 2.05]),
    )

    assert labels.tolist() == [0, 1]


def test_effective_router_fit_weights_use_official_macro_mid_mass_and_regret() -> None:
    """Catch a router objective that optimizes unweighted winner accuracy."""

    helper = getattr(nested_router, "effective_router_fit_weights", None)
    assert callable(helper)

    # The first row is a high official-mass, large-regret error; a row-accuracy
    # objective must not be allowed to treat it like either low-cost row.
    weights = helper(
        official_macro_mid_row_weight=np.array([0.5, 0.25, 0.25]),
        raw_mid_loss_a5=np.array([10.0, 3.0, 4.0]),
        raw_mid_loss_c2f=np.array([2.0, 1.0, 4.0]),
    )

    assert np.allclose(weights, np.array([4.0, 0.5, 0.0]))


def test_router_fit_contract_requires_effective_sample_weights() -> None:
    """Catch silent fallback to row-accuracy fitting without MiD-aligned weights."""

    assert "effective_sample_weights" in inspect.signature(CausalExpertRouter.fit).parameters


def test_router_pipeline_does_not_multiply_effective_weights_by_class_balance() -> None:
    """Catch sklearn class balancing that would alter the official MiD objective."""

    router = CausalExpertRouter(seed=7)

    assert router.pipeline.named_steps["router"].class_weight is None


def test_scaler_is_fit_only_on_inner_oof_rows() -> None:
    frame, inner_folds = _inner_oof_frame()
    fitted = fit_router_from_inner_oof(
        frame, inner_folds=inner_folds, outer_dev_sequences=OUTER_FOLDS[0], seed=7
    )
    expected = frame.loc[:, ROUTER_FEATURES].to_numpy(dtype=np.float64).mean(axis=0)
    actual = fitted.router.pipeline.named_steps["scale"].mean_

    assert np.allclose(actual, expected)
    assert fitted.router.signature.payload["fit_rows"] == len(frame)
    signature = fitted.router.signature.payload
    assert signature["fit_effective_sample_weights_sha256"]
    assert signature["fit_effective_sample_weight_sum"] > 0.0
    assert signature["sklearn"]["class_weight"] is None
    assert signature["fit_semantics"]["effective_sample_weight"] == (
        "official_macro_mid_row_weight * abs(raw_mid_loss_c2f - raw_mid_loss_a5)"
    )
    assert fitted.outer_dev_sequences == OUTER_FOLDS[0]


def test_router_is_deterministic_and_hard_routes_at_fixed_threshold() -> None:
    frame, inner_folds = _inner_oof_frame()
    first = fit_router_from_inner_oof(
        frame, inner_folds=inner_folds, outer_dev_sequences=OUTER_FOLDS[0], seed=7
    )
    second = fit_router_from_inner_oof(
        frame, inner_folds=inner_folds, outer_dev_sequences=OUTER_FOLDS[0], seed=7
    )
    features = frame.loc[:, ROUTER_FEATURES]

    assert first.router.signature.artifact_sha256 == second.router.signature.artifact_sha256
    assert json.loads(json.dumps(first.router.signature.payload))["artifact_sha256"] == (
        first.router.signature.artifact_sha256
    )
    assert np.array_equal(first.router.choose_c2f(features), second.router.choose_c2f(features))
    routed, choice, probability = first.router.route(
        features,
        a5_prediction_ttc_s=frame["a5_prediction_ttc_s"].to_numpy(),
        c2f_prediction_ttc_s=frame["c2f_prediction_ttc_s"].to_numpy(),
    )
    assert np.array_equal(choice, probability >= 0.5)
    assert np.array_equal(
        routed,
        np.where(choice, frame["c2f_prediction_ttc_s"], frame["a5_prediction_ttc_s"]),
    )


def test_degenerate_router_labels_fail_with_actionable_error() -> None:
    features = _features(rows=4)
    router = CausalExpertRouter(seed=7)

    with pytest.raises(RouterFitError, match="degenerate"):
        router.fit(
            features,
            np.zeros(4, dtype=np.int64),
            sample_tokens=("a", "b", "c", "d"),
            effective_sample_weights=np.ones(4, dtype=np.float64),
        )


def test_router_fit_rejects_zero_effective_mass_for_an_outcome() -> None:
    """Catch a fit that accepts a formally present but zero-cost class."""

    router = CausalExpertRouter(seed=7)
    with pytest.raises(RouterFitError, match="positive mass.*C2F-win"):
        router.fit(
            _features(rows=4),
            np.array([0, 0, 1, 1], dtype=np.int64),
            sample_tokens=("a", "b", "c", "d"),
            effective_sample_weights=np.array([1.0, 1.0, 0.0, 0.0]),
        )


def test_routing_point_ttc_rejects_selective_nan_and_requires_finite_point() -> None:
    trainer = pd.DataFrame(
        {
            "sample_token": ["token-a", "token-b"],
            "prediction_ttc_s": [np.nan, 1.0],
            "point_prediction_ttc_s": [2.5, 1.0],
        }
    )
    np.testing.assert_allclose(routing_point_ttc(trainer), [2.5, 1.0])
    missing = trainer.drop(columns=["point_prediction_ttc_s"])
    with pytest.raises(NestedRouterIntegrityError, match="point_prediction_ttc_s"):
        routing_point_ttc(missing)
    nonfinite = trainer.assign(point_prediction_ttc_s=[np.nan, 1.0])
    with pytest.raises(NestedRouterIntegrityError, match="non-finite"):
        routing_point_ttc(nonfinite)


def test_bind_expert_oof_replaces_selective_nan_with_trainer_point() -> None:
    oof = pd.DataFrame(
        {
            "token_id": ["token-b", "token-a"],
            "prediction_ttc": [1.0, np.nan],
            "finite": [True, True],
        }
    )
    trainer = pd.DataFrame(
        {
            "sample_token": ["token-a", "token-b"],
            "prediction_ttc_s": [np.nan, 1.0],
            "point_prediction_ttc_s": [2.5, 1.0],
        }
    )
    bound = bind_expert_oof_to_trainer_point_ttc(oof, trainer)
    np.testing.assert_allclose(bound["prediction_ttc"].to_numpy(dtype=np.float64), [1.0, 2.5])
    assert bool(bound["finite"].all())
    mismatched = trainer.assign(sample_token=["token-a", "token-c"])
    with pytest.raises(NestedRouterIntegrityError, match="token identity"):
        bind_expert_oof_to_trainer_point_ttc(oof, mismatched)


def test_reconstructed_event_count_is_integral_for_oof_schema() -> None:
    counts = integral_event_count_from_reconstructed([191601.182728, 9.6, 10.0])
    np.testing.assert_array_equal(counts, np.array([191601, 10, 10], dtype=np.int64))
    with pytest.raises(NestedRouterIntegrityError, match="not finite"):
        integral_event_count_from_reconstructed([1.0, np.nan])
    with pytest.raises(NestedRouterIntegrityError, match="negative"):
        integral_event_count_from_reconstructed([-0.6])
