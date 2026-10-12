"""Prospective query, tracking and nearest-GT rules from the author clarification."""

import numpy as np
import pytest

from operational.author_evttc.inputs import choose_box, sampled_frames
from operational.author_evttc.score import MODELS, nearest_truth, summarize


def test_sampling_is_timestamp_grid_not_every_other_valid_annotation():
    clock = np.array([20, 50_020, 100_021, 150_023, 250_019, 300_022])
    assert sampled_frames(clock, 0, 310_000) == [0, 2, -1, 5]


def test_sampling_bounds_and_clock_guard():
    assert sampled_frames(np.array([20, 100_021, 200_022]), 30, 199_999) == [1]
    with pytest.raises(ValueError, match="strictly"):
        sampled_frames(np.array([20, 20]), 0, 100)


def test_tracking_uses_previous_center_and_annotation_order_ties():
    objects = [{"bbox": [100, 100, 110, 110]}, {"bbox": [0, 0, 10, 10]}]
    assert choose_box(objects, None)[0] == 0
    assert choose_box(objects, (4, 4))[0] == 1
    assert choose_box(objects, (55, 55))[0] == 0


@pytest.mark.parametrize("query, expected", [(1.01, 3.0), (1.09, 5.0), (0.95, 3.0)])
def test_gt_uses_nearest_not_interpolation(query, expected):
    table = np.array([[0, 1.0, 0, 0, 3], [1, 1.1, 0, 0, 5]], dtype=float)
    assert nearest_truth(table, query)[0] == expected


def test_nearest_invalid_gt_is_not_replaced_with_another_label():
    table = np.array([[0, 1.0, 0, 0, np.nan], [1, 1.01, 0, 0, 5]])
    assert nearest_truth(table, 1.0)[1] == "INVALID_NEAREST_GT"


def test_gt_outside_tolerance_is_unavailable():
    table = np.array([[0, 1.0, 0, 0, 3]], dtype=float)
    assert nearest_truth(table, 1.05001)[1] == "GT_TIMESTAMP_MISMATCH"


def test_failed_prediction_keeps_denominator_and_common_cohort_is_explicit():
    first: dict[str, object] = {name: 2.0 for name in MODELS}
    second: dict[str, object] = {name: 4.0 for name in MODELS}
    first.update(scenario="s", truth_ttc_seconds=2.0)
    second.update(scenario="s", truth_ttc_seconds=2.0)
    second["Garl_full"] = None
    result = summarize([first, second])
    own = result["own_available"]["Garl_full"]
    assert own["labelled"] == 2
    assert own["finite_paired"] == 1
    assert own["missing_prediction_on_label"] == 1
    assert result["own_available"]["F_TRUE"]["RTE_pct"] == 50.0
    assert result["common_all_models"]["F_TRUE"]["RTE_pct"] == 0.0
