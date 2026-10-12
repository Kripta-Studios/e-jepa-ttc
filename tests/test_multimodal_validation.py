"""Scientific boundary tests for the additive frozen evaluator."""

import numpy as np
import torch

from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi
from operational.multimodal_validation.garl import timing
from operational.multimodal_validation.geometry import causal_prior
from operational.multimodal_validation.traces import Recorder


def row(anchor=300000):
    return dict(anchor_us=anchor, windows_us=[[0, 100000], [100000, 200000], [200000, 300000]])


def test_temporal_rejects_v13_style_unsynced_history():
    assert not timing(row(), np.array([150000, 250000, 350000]))["available"]


def test_temporal_wait_is_explicit_and_bounded():
    value = timing(row(), np.array([100000, 200050, 300050]))
    assert value["available"] and value["available_us"] == 300050
    assert not timing(row(), np.array([100000, 201001, 301001]))["available"]


def test_interval_rejected_even_when_individual_frames_within_tolerance():
    assert not timing(row(), np.array([100000, 199001, 300999]))["available"]


def test_channelwise_crop_equals_native_concatenated_crop():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(3, 24, 32)).astype(np.float32)
    b = rng.normal(size=a.shape).astype(np.float32)
    box = (-4, -2, 25, 27)
    ref = official_resize_roi(np.concatenate((a, b)), box, (128, 128)).numpy()
    actual = np.concatenate([official_resize_roi(x, box, (128, 128)).numpy() for x in (a, b)])
    assert np.array_equal(ref, actual)


def test_hooks_preserve_prediction_and_are_removed():
    model = torch.nn.Linear(3, 2).eval()
    data = torch.tensor([[1.0, 2.0, 3.0]])
    rec = Recorder()
    expected = model(data)
    result = rec.call(model, data)
    assert torch.equal(result, expected) and rec.rows[0]["bitwise_parity"]
    assert not model._forward_hooks


def test_prior_requires_past_support_and_drift():
    current = dict(shift_px=[2, 3], support_pixels=150, half_window_disagreement_px=1.0)
    assert not causal_prior(row(), None, current)["eligible"]
    old = dict(query_id="a", query_us=200000, shift_px=[1, 2])
    assert causal_prior(row(), old, current)["eligible"]
    assert not causal_prior(row(500001), old, current)["eligible"]
    assert not causal_prior(row(), old, dict(current, support_pixels=2))["eligible"]
