from __future__ import annotations

import torch

from e_jepa_ttc.data.synthetic_causal_scale import (
    SyntheticCausalScaleConfig,
    SyntheticCausalScaleDataset,
    synthetic_scale_config_identity,
)
from e_jepa_ttc.models.causal_scale_ttc import log_ratio_to_inverse_ttc


def _dataset(seed: int, samples: int = 32) -> SyntheticCausalScaleDataset:
    return SyntheticCausalScaleDataset(
        SyntheticCausalScaleConfig(
            samples=samples,
            seed=seed,
            canvas_size=48,
            background_events_per_endpoint=2,
            empty_probability=0.125,
        )
    )


def test_synthetic_causal_sample_is_deterministic_and_matches_event_contract() -> None:
    first = _dataset(101)[3]
    second = _dataset(101)[3]
    assert first["sample_id"] == "synthetic-101-3"
    for key in (
        "inputs",
        "delta_t_s",
        "target_ttc_seconds",
        "target_log_ratio",
        "target_valid",
        "target_masks",
        "mask_valid",
        "direction",
    ):
        assert torch.equal(first[key], second[key])
    assert first["inputs"].shape == (3, 12, 48, 48)
    assert first["target_masks"].shape == (3, 1, 48, 48)
    assert first["delta_t_s"].shape == (2,)
    assert torch.isfinite(first["inputs"]).all()


def test_nonempty_targets_obey_rasterized_scale_ttc_identity() -> None:
    dataset = _dataset(202, samples=48)
    checked = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample["target_valid"]):
            continue
        ratio = sample["target_log_ratio"].reshape(1)
        delta = sample["delta_t_s"][-1:].clone()
        inverse = log_ratio_to_inverse_ttc(ratio, delta)
        target = sample["target_ttc_seconds"]
        assert torch.allclose(inverse, target.reciprocal().reshape(1), atol=1.0e-6)
        assert int(torch.sign(target).item()) == int(sample["direction"].item())
        checked += 1
    assert checked >= 32


def test_seeded_groups_are_disjoint_and_cover_directions_and_empty_inputs() -> None:
    train = _dataset(101, samples=96)
    validation = _dataset(202, samples=96)
    train_ids = {str(train[index]["sample_id"]) for index in range(len(train))}
    validation_ids = {str(validation[index]["sample_id"]) for index in range(len(validation))}
    assert train_ids.isdisjoint(validation_ids)
    directions = {int(train[index]["direction"].item()) for index in range(len(train))}
    assert directions == {-1, 0, 1}
    empty = next(train[index] for index in range(len(train)) if train[index]["shape"] == "empty")
    assert torch.count_nonzero(empty["inputs"]) == 0
    assert torch.count_nonzero(empty["target_masks"]) == 0
    assert not bool(empty["target_valid"])


def test_synthetic_config_identity_binds_seed_and_sample_count() -> None:
    first = SyntheticCausalScaleConfig(seed=7, samples=16)
    second = SyntheticCausalScaleConfig(seed=8, samples=16)
    assert synthetic_scale_config_identity(first) != synthetic_scale_config_identity(second)
