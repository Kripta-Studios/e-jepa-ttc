"""V8 fold CSVs must record A5 unknown-support NaNs with a failure reason."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from e_jepa_ttc.evaluation.scientific_recovery_v8 import validate_oof_frame
from e_jepa_ttc.training.scientific_recovery_v8_trainer import export_v8_point_predictions


def test_unknown_support_nan_is_exported_with_failure_reason() -> None:
    preds, variances, finite, reasons = export_v8_point_predictions(
        [1.5, float("nan")],
        [0.1, 0.2],
        [True, False],
    )
    assert preds[0] == pytest.approx(1.5)
    assert math.isnan(preds[1])
    assert math.isnan(variances[1])
    assert finite == [True, False]
    assert reasons == ["", "no_known_causal_support"]


def test_known_non_finite_point_ttc_is_not_relabeled_as_unknown_support() -> None:
    _, _, finite, reasons = export_v8_point_predictions(
        [float("nan")],
        [0.1],
        [True],
    )
    assert finite == [False]
    assert reasons == ["non_finite_point_ttc"]


def test_oof_frame_with_documented_nan_validates() -> None:
    preds, variances, finite, reasons = export_v8_point_predictions(
        [2.0, float("nan")],
        [0.0, 0.0],
        [True, False],
    )
    frame = pd.DataFrame(
        {
            "token_id": ["a", "b"],
            "sequence_id": ["s0", "s0"],
            "track_id": ["t0", "t1"],
            "outer_fold": [0, 0],
            "seed": [7, 7],
            "target_ttc": [1.0, 2.0],
            "sample_weight": [1.0, 1.0],
            "prediction_ttc": preds,
            "prediction_log_variance": variances,
            "finite": finite,
            "failure_reason": reasons,
            "event_count": [10, 10],
            "event_rate": [1.0, 1.0],
            "support_ms": [100.0, 100.0],
            "model_name": ["timevol20_3", "timevol20_3"],
            "config_sha256": ["a" * 64, "a" * 64],
            "checkpoint_sha256": ["b" * 64, "b" * 64],
        }
    )
    validated = validate_oof_frame(frame, label="test")
    assert bool(validated["finite"].iloc[0]) is True
    assert bool(validated["finite"].iloc[1]) is False
