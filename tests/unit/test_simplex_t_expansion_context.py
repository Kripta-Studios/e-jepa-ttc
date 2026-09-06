"""D1 index membership depends on input windows, not targets or labelled past pairs."""

import copy

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.expansion_context import expansion_context_arrays


def fixture():
    rows = [
        {
            "sample_token": "q",
            "sequence_id": "s",
            "ttc": 1.0,
            "event_windows_us": [[200_000, 300_000], [300_000, 400_000]],
            "boxes_xyxy": [[0, 0, 8, 8], [1, 1, 9, 9]],
        }
    ]
    exposure = {"q": {"anchor_us": 400_000, "selected_exposure_end_us": 405_000}}
    return rows, exposure, {"s": (0, 1_000_000)}


def test_target_perturbation_leaves_all_indices_unchanged():
    rows, exposures, bounds = fixture()
    before = expansion_context_arrays(rows, exposures, bounds)
    rows[0].update(ttc=float("nan"), depth=-999, velocity=1e100, future_track_length=0)
    after = expansion_context_arrays(rows, exposures, bounds)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["valid"].sum() == 3
    assert before["valid"][0, -1]
    assert before["roi_available_us"][0] == 405_000


@pytest.mark.parametrize("failure", ["duplicate", "exposure", "bounds", "window"])
def test_invalid_input_contract_fails(failure):
    rows, exposures, bounds = fixture()
    if failure == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif failure == "exposure":
        exposures["q"]["selected_exposure_end_us"] = 399_999
    elif failure == "bounds":
        bounds["s"] = (100_001, 1_000_000)
    else:
        rows[0]["event_windows_us"] = [[200_000.0, 300_000.0], [300_000.0, 400_000.0]]
    with pytest.raises(ValueError):
        expansion_context_arrays(rows, exposures, bounds)
