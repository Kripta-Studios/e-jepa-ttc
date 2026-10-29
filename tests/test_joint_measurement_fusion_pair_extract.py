"""Pair extraction preserves the sensor clock and cannot silently use a live producer."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.joint_measurement.fusion_pair_extract import (
    FEATURE_NAMES,
    clock_pairs,
    extract_pair,
)


class PairFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.03))

    def forward(self, inputs, delta):
        descriptor = torch.cat(
            (inputs.new_zeros(len(inputs), 192), delta[:, None], delta.log()[:, None]), 1
        )
        raw = inputs.mean((1, 2, 3, 4)) * self.offset
        return {"descriptor": descriptor, "raw_phase": raw}


def rows():
    return pd.DataFrame(
        {
            "producer_times_us": [np.array([100000, 200000, 300000], np.int64)],
            "producer_sensor_times_us": [np.array([150000, 250002, 350005], np.int64)],
        }
    )


def test_distinct_native_clocks_and_last_pair_are_preserved():
    event = clock_pairs(
        rows(), modality="event", delta_t_s=np.array([0.1], np.float32), available=np.array([True])
    )
    rgb = clock_pairs(
        rows(),
        modality="rgb",
        delta_t_s=np.array([0.100003], np.float32),
        available=np.array([True]),
    )
    np.testing.assert_array_equal(event, [[200000, 300000]])
    np.testing.assert_array_equal(rgb, [[250002, 350005]])
    with pytest.raises(ValueError, match="differs"):
        clock_pairs(
            rows(),
            modality="rgb",
            delta_t_s=np.array([0.1], np.float32),
            available=np.array([True]),
        )


def test_unavailable_rgb_clock_is_masked_but_event_cannot_be_missing():
    data = pd.DataFrame({"producer_sensor_times_us": [np.array([], np.int64)]})
    values = clock_pairs(
        data, modality="rgb", delta_t_s=np.array([np.nan], np.float32), available=np.array([False])
    )
    np.testing.assert_array_equal(values, [[0, 0]])
    with pytest.raises(ValueError, match="fallback clock"):
        clock_pairs(
            data,
            modality="event",
            delta_t_s=np.array([np.nan], np.float32),
            available=np.array([False]),
        )


@pytest.mark.parametrize(
    "times", [np.array([1, 1], np.int64), np.array([2, 1], np.int64), np.array([1.0, 2.0])]
)
def test_invalid_native_clock_cannot_be_repaired(times):
    data = pd.DataFrame({"producer_times_us": [times]})
    with pytest.raises(ValueError):
        clock_pairs(
            data,
            modality="event",
            delta_t_s=np.array([0.1], np.float32),
            available=np.array([True]),
        )


def test_features_are_label_free_and_match_native_prediction():
    model = PairFixture().eval().requires_grad_(False)
    pixels = torch.ones(2, 2, 3, 4, 4)
    pixels[1] *= -1
    delta = torch.tensor([0.1, 0.100002])
    features, prediction = extract_pair(model, pixels, delta)
    assert features.shape == (2, len(FEATURE_NAMES)) == (2, 196)
    assert len(set(FEATURE_NAMES)) == 196
    np.testing.assert_array_equal(
        prediction, phase_to_ttc(model(pixels, delta)["raw_phase"]).numpy()
    )
    np.testing.assert_array_equal(features[:, 192], delta.numpy())
    assert prediction[0] > 0 and prediction[1] < 0
    assert model.offset.grad is None


@pytest.mark.parametrize("state", ["train", "requires_grad"])
def test_live_producer_is_rejected(state):
    model = PairFixture().eval().requires_grad_(False)
    if state == "train":
        model.train()
    else:
        model.requires_grad_(True)
    with pytest.raises(ValueError, match="private frozen"):
        extract_pair(model, torch.ones(1, 2, 3, 4, 4), torch.tensor([0.1]))
