"""Known displacement signs, support and flat-feature ties in the CPU diagnostic."""

import pytest
import torch

from operational.joint_measurement.correspondence_diagnostic import OFFSETS, matching, translate


@pytest.mark.parametrize("offset", OFFSETS)
def test_known_translation_sign_and_exact_interior_match(offset):
    source = torch.randn(1, 64, 32, 32, generator=torch.Generator().manual_seed(19))
    current = translate(source, *offset)
    result = matching(source, current, *offset)
    assert result["positions"] == 256
    assert result["strict_correct"] == 256
    assert result["strict_accuracy"] == 1
    assert result["mean_offset_error_cells"] == 0


def test_flat_feature_ties_are_not_correspondence_evidence():
    source = torch.ones(1, 64, 32, 32)
    assert matching(source, source, -2, -2)["argmax_accuracy"] == 1
    assert matching(source, source, -2, -2)["strict_accuracy"] == 0


def test_translation_does_not_wrap_pixels():
    source = torch.arange(25).reshape(1, 1, 5, 5)
    result = translate(source, 1, -1)
    assert not result[..., 0, :].any()
    assert not result[..., :, -1].any()
    assert torch.equal(result[..., 1:, :-1], source[..., :-1, 1:])


def test_random_spatial_permutation_is_not_an_exact_translation():
    source = torch.randn(1, 64, 32, 32, generator=torch.Generator().manual_seed(19))
    permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
    current = source.flatten(2)[:, :, permutation].reshape_as(source)
    assert matching(source, current, 2, 0)["strict_accuracy"] < 0.15
