from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import operational.evttc_rgb_transfer.model as module


class _Full(nn.Module):
    dT = 0.1  # noqa: N815 - pinned upstream API

    def forward_test(self, sensor: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert sensor.shape == (1, 46, 128, 128)
        return sensor.new_tensor([[4.0, 5.0]]), None


def _runtime() -> module.FullGarl:
    runtime = module.FullGarl.__new__(module.FullGarl)
    runtime.device = torch.device("cpu")
    runtime.model = _Full()
    return runtime


def test_compose_sensor_preserves_release_channel_order() -> None:
    rgb = np.empty((2, 3, 128, 128), dtype=np.float32)
    events = np.empty((2, 20, 128, 128), dtype=np.float32)
    for endpoint in range(2):
        for channel in range(3):
            rgb[endpoint, channel] = 10 * endpoint + channel
        for channel in range(20):
            events[endpoint, channel] = 100 * endpoint + channel

    sensor = module.FullGarl.compose_sensor(rgb, events)
    assert sensor.shape == (46, 128, 128)
    expected = [*range(3), *range(10, 13), *range(20), *range(100, 120)]
    np.testing.assert_array_equal(sensor[:, 0, 0], expected)


def test_predict_uses_native_unclipped_height_formula() -> None:
    runtime = _runtime()
    rgb = np.zeros((2, 3, 128, 128), dtype=np.float32)
    events = np.zeros((2, 20, 128, 128), dtype=np.float32)
    result = runtime.predict(rgb, events)
    assert result["heights"] == [4.0, 5.0]
    assert result["ttc"] == pytest.approx(0.5)


def test_release_formula_preserves_singularity() -> None:
    output = module.release_ttc(np.asarray([[2.0, 2.0]], np.float32), 0.1)
    assert np.isinf(output[0])


@pytest.mark.parametrize(
    "rgb,events,match",
    [
        (
            np.zeros((2, 3, 128, 128), np.float64),
            np.zeros((2, 20, 128, 128), np.float32),
            "rgb_endpoints",
        ),
        (
            np.zeros((1, 3, 128, 128), np.float32),
            np.zeros((2, 20, 128, 128), np.float32),
            "rgb_endpoints",
        ),
        (
            np.zeros((2, 3, 128, 128), np.float32),
            np.zeros((2, 19, 128, 128), np.float32),
            "event_endpoints",
        ),
    ],
)
def test_compose_rejects_noncanonical_inputs(
    rgb: np.ndarray, events: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        module.FullGarl.compose_sensor(rgb, events)


def test_config_contract_rejects_nearby_ablation() -> None:
    configuration = {
        "dataset": {
            "mode": "image_event",
            "sync": "front",
            "window_interval": 1,
            "img_mean": list(module._RGB_MEAN),
            "img_std": list(module._RGB_STD),
        },
        "model": {
            "fusion_style": "early_fusion",
            "mode": "height_ratio",
            "input_feat_num_rgb": 6,
            "input_feat_num_event": 40,
            "input_feat_size": [128, 128],
            "frame_num": 2,
            "with_decoder": True,
            "num_layers": 50,
        },
    }
    with pytest.raises(ValueError, match="fusion_style"):
        module._validate_config(configuration)


def test_expected_public_paths_are_stable() -> None:
    public = Path("public_garl")
    assert public / module.CHECKPOINT_NAME == public / "paper_ours_full.pth"
    assert public / module.CONFIG_RELATIVE == public / "configs/ablation/ours_full.yaml"
    assert module.CHECKPOINT_BYTES == 973_290_993
