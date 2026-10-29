from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.analyze_v5_a8_oof_failure_modes import (
    FAMILY_FEATURES,
    align_predictions,
    feature_association,
    raw_mid_per_sample,
    select_family,
)


def _predictions(label: str, prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_token": ["a", "b"],
            "sequence_id": ["s0", "s1"],
            "track_id": ["t0", "t1"],
            "target_ttc_s": [2.0, -2.0],
            "prediction_ttc_s": prediction,
            "ignored": label,
        }
    )


def test_raw_mid_matches_zero_for_exact_prediction_and_preserves_invalid() -> None:
    values = raw_mid_per_sample([2.0, -2.0, 2.0], [2.0, -2.0, float("nan")])

    assert values[:2].tolist() == [0.0, 0.0]
    assert math.isnan(values[2])


def test_align_predictions_preserves_failures_and_exact_identities() -> None:
    result = align_predictions(
        {
            "a6": _predictions("a6", [1.0, float("nan")]),
            "a8_0": _predictions("a8", [2.0, -2.0]),
            "garl": _predictions("garl", [2.1, -1.9]),
        }
    )

    assert len(result) == 2
    assert math.isnan(result.loc[1, "a6_prediction"])
    assert result.loc[1, "a8_0_prediction"] == -2.0


def test_feature_association_uses_sequence_robust_effect() -> None:
    rows = []
    for sequence in ("s0", "s1", "s2"):
        for value in range(12):
            rows.append(
                {
                    "sequence_id": sequence,
                    "feature": float(value),
                    "outcome": -float(value),
                }
            )
    result = feature_association(pd.DataFrame(rows), "feature", "outcome")

    assert result["global_spearman"] == -1.0
    assert result["median_absolute_sequence_spearman"] == 1.0
    assert result["sequence_sign_agreement"] == 1.0


def test_select_family_prefers_low_complexity_family_within_near_tie() -> None:
    associations = {}
    for features in FAMILY_FEATURES.values():
        for feature in features:
            associations[feature] = {
                "median_absolute_sequence_spearman": 0.05,
                "sequence_sign_agreement": 1.0,
            }
    associations["a8_transport_flow_magnitude"] = {
        "median_absolute_sequence_spearman": 0.50,
        "sequence_sign_agreement": 0.8,
    }
    associations["a8_transport_entropy"] = {
        "median_absolute_sequence_spearman": 0.46,
        "sequence_sign_agreement": 0.8,
    }

    result = select_family(associations)

    assert result["selected_family"] == "transport_confidence"
    assert result["selected_branch"] == "V6.1_CONFIDENCE_AWARE_FUSION"


def test_select_family_rethinks_objective_when_no_family_is_robust() -> None:
    associations = {
        feature: {
            "median_absolute_sequence_spearman": np.nan,
            "sequence_sign_agreement": 0.0,
        }
        for features in FAMILY_FEATURES.values()
        for feature in features
    }

    result = select_family(associations)

    assert result["selected_family"] is None
    assert result["selected_branch"] == "V6.1_RETHINK_TRANSPORT_OBJECTIVE"
