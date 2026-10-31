"""Correlated failure simulation is deterministic and cannot mutate clean experts."""

import inspect

import torch

from scripts.train_phase_correlated_fusion import corrupt_native, valid_training_rows


def test_corruption_is_deterministic_and_does_not_alias_clean_input():
    native = torch.arange(1, 12001, dtype=torch.float32).reshape(-1, 3)
    original = native.clone()
    left = corrupt_native(native, generator=torch.Generator().manual_seed(7))
    right = corrupt_native(native, generator=torch.Generator().manual_seed(7))
    assert torch.equal(left, right)
    assert torch.equal(native, original)
    assert left.data_ptr() != native.data_ptr()
    left[0, 0] = -999
    assert torch.equal(native, original)


def test_corruption_includes_paired_triple_and_single_patterns_with_shared_factor():
    native = torch.ones((8192, 3))
    corrupted = corrupt_native(native, generator=torch.Generator().manual_seed(13))
    changed = corrupted != native
    patterns = {tuple(row.nonzero().flatten().tolist()) for row in changed}
    assert patterns == {(), (0,), (1,), (2,), (0, 1), (0, 2), (0, 1, 2)}
    assert 0.22 < float(changed.any(1).float().mean()) < 0.28
    for row, mask in zip(corrupted, changed, strict=True):
        if mask.any():
            assert torch.unique(row[mask]).numel() == 1
            assert float(row[mask][0]) in {-3.0, -1.0, 0.25, 4.0}


def test_corruption_interface_cannot_receive_geometry_quality_or_targets():
    assert set(inspect.signature(corrupt_native).parameters) == {"native", "generator"}


def test_training_admits_geometry_only_query_as_original_iq_does():
    native = torch.full((3, 3), float("nan"))
    geometry = torch.tensor([2.0, float("nan"), 2.0])
    target = torch.tensor([2.0, 2.0, float("nan")])
    assert valid_training_rows(native, geometry, target).tolist() == [True, False, False]
