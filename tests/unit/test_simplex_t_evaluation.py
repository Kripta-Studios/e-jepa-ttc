"""Known-value metrics and factorial interaction algebra."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

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
