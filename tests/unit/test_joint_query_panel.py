"""Adversarial query admission and paired outcome arithmetic."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.query_panel import (
    MODELS,
    join_queries,
    summarize,
    validate_clock_bindings,
)


@pytest.fixture
def tables():
    metadata = pd.DataFrame(
        {
            "sample_token": ["a", "b", "c"],
            "group_id": ["g"] * 3,
            "sequence_id": ["s"] * 3,
            "role": ["P"] * 3,
            "target_ttc": [-2.0, -4.0, 1.0],
            "query_time_us": [1000, 2000, 3000],
            "cutoff_us": [1000, 2000, 3000],
            "event_previous_start_us": [10, 20, 30],
            "event_previous_end_us": [11, 21, 31],
            "event_current_start_us": [11, 21, 31],
            "event_current_end_us": [12, 22, 32],
            "native_delta_t_s": [0.1] * 3,
            "metric_delta_t_s": [0.1] * 3,
        }
    )
    predictions = {}
    for model, pred in {
        "A5": [-2.0, 4.0, 2.0],
        "NATIVE": [2.0, -4.0, 3.0],
        "CORR": [2.0, 4.0, 1.0],
        "C1": [-2.0, -4.0, 1.5],
    }.items():
        frame = metadata[["sample_token", "group_id", "sequence_id", "role", "target_ttc"]].copy()
        frame["prediction_ttc"] = pred
        frame["raw_phase"] = -np.log1p(-0.1 / np.asarray(pred))
        predictions[model] = frame.iloc[::-1].copy()
    return metadata, predictions


def test_keyed_alignment_and_complementarity(tables):
    metadata, predictions = tables
    panel = join_queries(metadata, predictions)
    assert panel.sample_token.tolist() == ["a", "b", "c"]
    assert panel.negative_CORR_sign_error.tolist() == [True, True, False]
    assert panel.crucial_CORR_beats_A5_and_NATIVE.tolist() == [False, False, True]
    summary = summarize(panel, "P")
    a5 = next(
        r for r in summary if r["subset"] == "negative_CORR_sign_error" and r["model"] == "A5"
    )
    assert a5["rows"] == 2 and a5["sign_correct"] == 1
    critical = next(
        r
        for r in summary
        if r["subset"] == "crucial_CORR_beats_A5_and_NATIVE" and r["model"] == "NATIVE"
    )
    assert critical["mean_delta_AE_s"] == 2.0
    assert critical["mean_delta_MiD"] == pytest.approx(
        10000 * abs(np.log1p(-0.1 / 3) - np.log1p(-0.1))
    )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra", "target", "group", "role", "clock"]
)
def test_reject_incompatible_predictions(tables, mutation):
    metadata, predictions = tables
    frame = predictions["CORR"].copy()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "extra":
        frame.loc[frame.index[0], "sample_token"] = "different"
    elif mutation == "target":
        frame.loc[frame.index[0], "target_ttc"] += 0.001
    elif mutation == "group":
        frame.loc[frame.index[0], "group_id"] = "another"
    elif mutation == "role":
        frame["role"] = "H"
    else:
        frame["query_time_us"] = 0
    predictions["CORR"] = frame
    with pytest.raises(ValueError):
        join_queries(metadata, predictions)


def test_reject_duplicate_metadata_and_overlapping_clock(tables):
    metadata, predictions = tables
    with pytest.raises(ValueError, match="Duplicate"):
        join_queries(pd.concat([metadata, metadata.iloc[:1]]), predictions)
    metadata.loc[0, "event_current_start_us"] = 10
    with pytest.raises(ValueError, match="clock"):
        join_queries(metadata, predictions)


def test_binding_rejects_same_shape_but_different_sensor_clock():
    identities = {
        m: {"role": "P", "fit_id": value[0], "sensor_complete": "same"}
        for m, value in MODELS.items()
    }
    validate_clock_bindings(identities, "P", "same")
    identities["C1"]["sensor_complete"] = "other_clock"
    with pytest.raises(ValueError, match="clock"):
        validate_clock_bindings(identities, "P", "same")


def test_null_group_rejected_before_group_summary(tables):
    metadata, predictions = tables
    metadata.loc[0, "group_id"] = None
    with pytest.raises(ValueError, match="Null group"):
        join_queries(metadata, predictions)


def test_invalid_prediction_retained_and_summary_not_filtered(tables):
    metadata, predictions = tables
    predictions["CORR"].loc[0, "prediction_ttc"] = np.nan
    panel = join_queries(metadata, predictions)
    assert len(panel) == 3
    summary = summarize(panel, "P")
    negative = next(r for r in summary if r["subset"] == "negative_all" and r["model"] == "CORR")
    assert negative["rows"] == 2 and negative["invalid"] == 1
    assert negative["mean_MiD"] is None and negative["mean_AE_s"] is None
