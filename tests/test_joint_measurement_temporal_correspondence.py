"""Reciprocal matching must distinguish stationary, shifted, flat and missing support."""

import pytest
import torch

from operational.joint_measurement.correspondence_diagnostic import translate
from operational.joint_measurement.temporal_correspondence import coherence


def example():
    source = torch.randn(1, 64, 32, 32, generator=torch.Generator().manual_seed(19))
    support = torch.zeros(32, 32, dtype=torch.bool)
    support[4:-4, 4:-4] = True
    return source, support


@pytest.mark.parametrize("dy,dx", [(2, 0), (0, -2), (-1, 1)])
def test_known_motion_is_reciprocal_and_nonzero(dy, dx):
    source, support = example()
    result = coherence(source, translate(source, dy, dx), support)
    assert result["strict_mutual_fraction"] == 1
    assert result["strict_mutual_nonzero_fraction"] == 1
    assert result["mean_cycle_error_cells"] == 0
    assert result["displacement_counts"][(dy + 2) * 5 + dx + 2] == 576


def test_stationary_texture_cannot_supply_nonzero_motion_evidence():
    source, support = example()
    result = coherence(source, source, support)
    assert result["strict_mutual_fraction"] == 1
    assert result["strict_mutual_nonzero_fraction"] == 0


def test_collapsed_maps_do_not_count_as_reciprocal_evidence():
    source, support = example()
    source.fill_(1)
    result = coherence(source, source, support)
    assert result["strict_mutual"] == 0
    assert result["mean_top_margin"] == 0


def test_empty_foreground_is_explicitly_missing_not_perfect_or_zero_error():
    source, support = example()
    support.fill_(False)
    result = coherence(source, source, support)
    assert result["positions"] == 0
    assert result["strict_mutual_fraction"] is None
    assert result["mean_cycle_error_cells"] is None


def test_permuted_features_reduce_consistency_on_known_texture():
    source, support = example()
    permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
    null = source.flatten(2)[:, :, permutation].reshape_as(source)
    assert coherence(source, null, support)["strict_mutual_fraction"] < 0.75
