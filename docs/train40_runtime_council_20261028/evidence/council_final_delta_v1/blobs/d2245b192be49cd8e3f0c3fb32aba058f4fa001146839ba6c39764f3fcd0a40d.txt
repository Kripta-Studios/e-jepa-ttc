"""Independent sorted-quantile checks for exact event normalization selection."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from operational.train40_runtime.normalization import ExactTopKNormalizer, percentile95_tail_size


def canonical(voxel):
    occupied = voxel > 1e-6
    nonzero = occupied.sum(1)
    values = voxel.sort(dim=1).values
    rank = (nonzero - 1).clamp_min(0).double() * 0.95
    lower, upper = rank.floor().long(), rank.ceil().long()
    offset = voxel.shape[1] - nonzero
    a = values.gather(1, (offset + lower).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
    b = values.gather(1, (offset + upper).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
    fraction = rank - lower
    scale = torch.where(fraction >= 0.5, b - (b - a) * (1.0 - fraction), a + (b - a) * fraction)
    normalized = voxel / scale.clamp_min(1e-6)[:, None]
    normalized.masked_fill_(~occupied, 0.0)
    return normalized, scale


@pytest.mark.parametrize("columns", [1, 2, 19, 20, 21, 99, 100, 101, 1001, 163840])
def test_tail_bound_contains_every_possible_occupied_rank(columns):
    occupancy = torch.arange(columns + 1, dtype=torch.int64)
    rank = (occupancy - 1).clamp_min(0).double() * 0.95
    from_max = (occupancy - 1 - rank.floor()).clamp_min(0)
    assert int(from_max.max()) < percentile95_tail_size(columns)


@pytest.mark.parametrize("columns", [1, 2, 19, 20, 21, 99, 101, 1001])
def test_all_occupancy_counts_and_interpolation_branches_are_bit_exact(columns):
    rng = np.random.default_rng(7)
    values = np.zeros((columns + 1, columns), dtype=np.float64)
    for occupancy in range(columns + 1):
        values[occupancy, :occupancy] = rng.lognormal(0.0, 4.0, occupancy)
        rng.shuffle(values[occupancy])
    original = torch.from_numpy(values)
    expected, expected_scale = canonical(original)
    proposed = original.clone()
    normalizer = ExactTopKNormalizer(columns + 1, columns, device="cpu")
    scale = normalizer.normalize_(proposed)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(proposed, expected, rtol=0, atol=0)
    torch.testing.assert_close(proposed.float(), expected.float(), rtol=0, atol=0)


def test_threshold_ties_empty_and_single_value_match_numpy():
    source = torch.tensor(
        [
            [0.0, 1e-7, 1e-6, -1e-10, 0.0, 0.0],
            [0.0, 1e-7, 1e-6, 1.00000001e-6, 0.0, 0.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [2.0, 5.0, 7.0, 17.0, 17.0, 17.0],
        ],
        dtype=torch.float64,
    )
    normalizer = ExactTopKNormalizer(4, 6, device="cpu")
    proposed = source.clone()
    scale = normalizer.normalize_(proposed)
    expected, expected_scale = canonical(source)
    torch.testing.assert_close(proposed, expected, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    for index in range(1, 4):
        occupied = source[index].numpy()
        occupied = occupied[occupied > 1e-6]
        assert scale[index] == np.percentile(occupied, 95)


def test_actual_six_window_shape_and_reused_smaller_batch():
    generator = torch.Generator().manual_seed(7)
    original = torch.rand((12, 163840), dtype=torch.float64, generator=generator)
    original[original < 0.7] = 0
    normalizer = ExactTopKNormalizer(12, 163840, device="cpu")
    assert normalizer.tail_size == 8193
    assert normalizer.allocated_bytes < 4 * 1024**2
    for rows in (12, 6, 1):
        sample = original[:rows].clone()
        expected, expected_scale = canonical(sample)
        scale = normalizer.normalize_(sample)
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_tail_size(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        percentile95_tail_size(invalid)


def test_shape_dtype_and_layout_contract():
    normalizer = ExactTopKNormalizer(2, 10, device="cpu")
    invalid = [
        torch.zeros((2, 10), dtype=torch.float32),
        torch.zeros((3, 10), dtype=torch.float64),
        torch.zeros((2, 11), dtype=torch.float64),
        torch.zeros((2, 20), dtype=torch.float64)[:, ::2],
    ]
    for value in invalid:
        with pytest.raises(ValueError, match="contiguous resident float64"):
            normalizer.normalize_(value)


@pytest.mark.skipif(os.getenv("RUN_TRAIN40_NORMALIZATION_CUDA") != "1", reason="explicit GPU lease")
def test_cuda_twenty_object_selection_is_bit_exact():
    generator = torch.Generator().manual_seed(7)
    original = torch.rand((120, 163840), dtype=torch.float64, generator=generator).cuda()
    original[original < 0.5] = 0
    expected, expected_scale = canonical(original)
    proposed = original.clone()
    normalizer = ExactTopKNormalizer(120, 163840, device="cuda")
    scale = normalizer.normalize_(proposed)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(proposed, expected, rtol=0, atol=0)
