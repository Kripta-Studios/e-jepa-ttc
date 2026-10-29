"""Historical batch reconstruction must use only the current query's inputs."""

import pytest
import torch

from operational.joint_measurement.fusion_cost_shape_probe import repeat_observation


def test_repeated_observation_preserves_pixels_clock_and_original() -> None:
    sensor = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    delta = torch.tensor([[0.1, 0.100002]], dtype=torch.float32)
    repeated, times = repeat_observation(sensor, delta, frames=3, batch=27)
    assert repeated.shape == (27, 3, 2, 2, 2)
    assert all(torch.equal(row, sensor[0]) for row in repeated)
    assert all(torch.equal(row, delta[0]) for row in times)
    repeated[0].zero_()
    assert sensor.sum() > 0


@pytest.mark.parametrize("frames,batch", [(2, 30), (3, 0), (3, 33)])
def test_unregistered_shape_is_rejected(frames, batch) -> None:
    with pytest.raises(ValueError):
        repeat_observation(torch.zeros(1, 3, 2, 2, 2), torch.ones(1, 2), frames=frames, batch=batch)


def test_precision_cannot_change() -> None:
    with pytest.raises(ValueError, match="FP32"):
        repeat_observation(
            torch.zeros(1, 3, 2, 2, 2, dtype=torch.float64), torch.ones(1, 2), frames=3, batch=32
        )


def test_clock_count_cannot_change() -> None:
    with pytest.raises(ValueError):
        repeat_observation(torch.zeros(1, 3, 2, 2, 2), torch.ones(1, 1), frames=3, batch=32)
