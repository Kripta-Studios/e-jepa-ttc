"""Known synthetic image scale must map coordinates in the declared direction."""

import pytest
import torch

from operational.joint_measurement.scale_correspondence_diagnostic import matching, warp


@pytest.mark.parametrize("scale", [0.9, 1.1])
def test_scale_warp_inverse_sampling(scale: float) -> None:
    x = torch.arange(32).float() + 0.5 - 16
    pixels = x[None, None, None].expand(1, 1, 32, 32)
    current = warp(pixels, scale)
    expected = (x / scale)[None, None, None].expand_as(current)
    torch.testing.assert_close(
        current[..., 8:-8, 8:-8], expected[..., 8:-8, 8:-8], atol=1e-5, rtol=0
    )


def test_matching_identity_and_constant_map_ties() -> None:
    features = torch.randn((1, 16, 32, 32), generator=torch.Generator().manual_seed(7))
    support = torch.zeros(32, 32, dtype=torch.bool)
    support[8:-8, 8:-8] = True
    result = matching(features, features, 1.0, support)
    assert result["strict_nearest_accuracy"] == 1
    assert result["mean_offset_error_cells"] == 0
    constant = torch.ones_like(features)
    assert matching(constant, constant, 1.1, support)["strict_nonzero_accuracy"] == 0
