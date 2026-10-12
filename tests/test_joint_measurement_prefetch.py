"""Speculative reads must preserve sample order, duplicates and exact resume RNG state."""

import torch

from operational.joint_measurement.prefetch import UniformPrefetch


def test_prefetch_does_not_reserve_or_advance_unused_draws():
    actual = torch.Generator().manual_seed(7)
    reference = torch.Generator().manual_seed(7)
    with UniformPrefetch(lambda ids: torch.tensor(ids), 5, 32, actual) as loader:
        assert torch.equal(actual.get_state(), reference.get_state())
        for _ in range(10):
            indices, values = loader.next()
            expected = torch.randint(5, (32,), generator=reference)
            assert indices == expected.tolist()
            assert torch.equal(values, expected)
            assert torch.equal(actual.get_state(), reference.get_state())
    assert torch.equal(actual.get_state(), reference.get_state())


def test_prefetch_restart_has_the_uninterrupted_sample_sequence():
    reference = torch.Generator().manual_seed(23)
    expected = [torch.randint(100, (32,), generator=reference).tolist() for _ in range(10)]
    sampler = torch.Generator().manual_seed(23)
    seen = []
    with UniformPrefetch(list, 100, 32, sampler) as loader:
        for _ in range(5):
            seen.append(loader.next()[0])
        checkpoint_state = sampler.get_state()
    restored = torch.Generator().set_state(checkpoint_state)
    with UniformPrefetch(list, 100, 32, restored) as loader:
        for _ in range(5):
            seen.append(loader.next()[0])
    assert seen == expected
    assert torch.equal(restored.get_state(), reference.get_state())
