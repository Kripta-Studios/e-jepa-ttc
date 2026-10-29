"""Regressions for I/O-efficient causal-scale cache sampling."""

from __future__ import annotations

import torch

from e_jepa_ttc.training.causal_scale_eap import ShardGroupedRandomSampler

GROUPS = ((0, 1, 2), (3, 4), (5, 6, 7, 8))


def _order(seed: int) -> list[int]:
    sampler = ShardGroupedRandomSampler(
        GROUPS,
        dataset_size=9,
        generator=torch.Generator().manual_seed(seed),
    )
    return list(sampler)


def test_shard_sampler_covers_once_and_keeps_groups_contiguous() -> None:
    order = _order(7)

    assert sorted(order) == list(range(9))
    for group in GROUPS:
        positions = sorted(order.index(index) for index in group)
        assert positions == list(range(positions[0], positions[0] + len(group)))


def test_shard_sampler_is_seed_deterministic() -> None:
    assert _order(7) == _order(7)
    assert _order(7) != _order(13)


def test_shard_sampler_rejects_incomplete_partition() -> None:
    try:
        ShardGroupedRandomSampler(
            ((0, 1), (3,)),
            dataset_size=4,
            generator=torch.Generator().manual_seed(7),
        )
    except ValueError as error:
        assert "partition" in str(error)
    else:
        raise AssertionError("incomplete shard partition must fail")
