"""Known-value metrics and factorial interaction algebra."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.simplex_t.evaluation import factorial_contrasts, prediction_frame


def test_factor_interactions_are_paired_differences():
    losses = {
        (d, h, c): np.array([10 + 2 * d + 3 * h + 5 * c + 7 * d * h])
        for d, h, c in product((0, 1), repeat=3)
    }
    effect = factorial_contrasts(losses)
    assert effect["D"][0] == 5.5
    assert effect["H"][0] == 6.5
    assert effect["C"][0] == 5
    assert effect["DxH"][0] == 7
    assert effect["DxC"][0] == effect["HxC"][0] == effect["DxHxC"][0] == 0


def test_incomplete_factorial_not_reported_as_data_effect():
    with pytest.raises(ValueError, match="complete"):
        factorial_contrasts({(0, 0, 0): np.array([1.0])})


def test_zero_phase_scores_emitted_positive60_not_zero():
    meta = pd.DataFrame(
        dict(sample_token=["q"], sequence_id=["s"], track_id=["t"], target_ttc=[60.0])
    )
    output = dict(
        point_phase=np.array([0.0]),
        raw_location=np.array([0.0]),
        raw_residual=np.array([0.0]),
        q10=np.array([-0.03]),
        q90=np.array([0.03]),
        relative_cost=np.zeros((1, 3)),
    )
    frame = prediction_frame(
        meta, np.full((1, 3), 10.0), output, np.array([[-1, 0]]), arm="fixture", seed=7, fold=0
    )
    assert frame.prediction_ttc_s.iloc[0] == 60
    assert frame.loss.iloc[0] == 0
    assert frame.cold_start.iloc[0]


def test_selector_exports_exact_original_ttc_without_residual_projection():
    expert = np.array([[80.1234567890123, -17.1234567890123, 2.25]])
    selected = expert[0, 0]
    metadata = pd.DataFrame(
        dict(sample_token=["q"], sequence_id=["s"], track_id=["t"], target_ttc=[selected])
    )
    output = dict(
        point_phase=benchmark_phase(np.array([selected])).astype(np.float32),
        raw_location=np.array([0.02]),
        raw_residual=np.array([0.0]),
        q10=np.array([-0.03]),
        q90=np.array([0.03]),
        relative_cost=np.array([[0.0, 1.0, 2.0]]),
    )
    frame = prediction_frame(
        metadata,
        expert,
        output,
        np.array([[0]]),
        arm="SELECTOR",
        seed=7,
        fold=0,
        output_mode="selector",
    )
    assert frame.prediction_ttc_s.iloc[0] == selected
    assert frame.loss.iloc[0] == 0
    assert frame.hull_position.iloc[0] == "inside"


def test_selector_preserves_infinite_original_and_exports_flag():
    metadata = pd.DataFrame(
        dict(sample_token=["q"], sequence_id=["s"], track_id=["t"], target_ttc=[2.0])
    )
    output = dict(
        point_phase=np.array([0.0]),
        raw_location=np.array([0.0]),
        raw_residual=np.array([0.0]),
        q10=np.array([-0.03]),
        q90=np.array([0.03]),
        relative_cost=np.array([[2.0, 1.0, 0.0]]),
    )
    frame = prediction_frame(
        metadata,
        np.array([[1.0, 2.0, np.inf]]),
        output,
        np.array([[0]]),
        arm="SELECTOR",
        seed=7,
        fold=0,
        output_mode="selector",
    )
    assert np.isposinf(frame.prediction_ttc_s.iloc[0])
    assert frame.prediction_phase.iloc[0] == 0
    assert frame.prediction_ttc_infinite.iloc[0]
    assert frame.expert2_ttc_infinite.iloc[0]
    assert not frame.finite_ttc_cap.iloc[0]
    assert np.isfinite(frame.loss.iloc[0])


def test_escape_gain_uses_emitted_current_median_baseline():
    metadata = pd.DataFrame(
        dict(sample_token=["q"], sequence_id=["s"], track_id=["t"], target_ttc=[60.0])
    )
    output = dict(
        point_phase=np.array([0.0]),
        raw_location=np.array([0.0]),
        raw_residual=np.array([0.0]),
        q10=np.array([-0.03]),
        q90=np.array([0.03]),
        relative_cost=np.zeros((1, 3)),
    )
    frame = prediction_frame(
        metadata,
        np.array([[80.0, 90.0, 100.0]]),
        output,
        np.array([[0]]),
        arm="TPR",
        seed=7,
        fold=0,
    )
    assert frame.gain_over_current_median.iloc[0] == 0
