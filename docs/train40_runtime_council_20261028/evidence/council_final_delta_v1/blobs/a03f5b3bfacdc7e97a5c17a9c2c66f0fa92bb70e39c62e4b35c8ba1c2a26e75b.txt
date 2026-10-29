"""Ensure failed native predictions cannot disappear from TRAIN comparison coverage."""

import numpy as np
import pytest

from operational.train40_system.comparison_metrics import evaluate
from operational.train40_system.garl_predictions import release_ttc


def test_perfect_signed_targets_have_zero_error_in_every_bucket():
    truth = np.asarray([-2.0, 1.0, 4.0, 8.0])
    result = evaluate(truth.copy(), truth)
    assert result["population"] == 4
    assert result["weighted_MiD_phase_valid"] == 0
    assert result["failure_rate"] == 0


def test_invalid_outputs_stay_in_bucket_denominators():
    result = evaluate(np.asarray([np.inf, 0.0, 4.0, 8.0]), np.asarray([-2.0, 1.0, 4.0, 8.0]))
    assert result["population"] == 4
    assert result["finite_coverage"] == 0.75
    assert result["failure_rate"] == 0.5
    assert result["weighted_MiD_phase_valid"] is None
    assert result["buckets"]["negative"]["population"] == 1
    assert result["buckets"]["negative"]["failure_rate"] == 1


def test_native_conversion_keeps_unclipped_infinite_and_negative_outputs():
    heights = np.asarray([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], np.float32)
    value = release_ttc(heights, 0.1)
    assert np.isinf(value[0])
    assert value[1] == np.float32(-0.1)
    assert value[2] == np.float32(0.2)


def test_physical_contact_relabelling_is_rejected():
    with pytest.raises(ValueError, match="original phase domain"):
        evaluate(np.asarray([1.0]), np.asarray([0.0]))
