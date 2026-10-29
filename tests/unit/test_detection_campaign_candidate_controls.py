from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from e_jepa_ttc.detection.campaign_candidate_controls import (
    CappedRareIdentitySampler,
    PhotometricPolicy,
    RareIdentityPolicy,
    StatefulPhotometricAugmenter,
)


def rare_policy() -> RareIdentityPolicy:
    return RareIdentityPolicy(
        rare_classes=("bicycle", "other_vehicle"),
        rare_draw_fraction=0.5,
        max_frame_exposures_per_epoch=4,
        max_identity_exposure_multiplier=4,
    )


def sampler() -> CappedRareIdentitySampler:
    tokens = tuple(f"train:{index}" for index in range(12))
    memberships = {
        "train:0": (("other_vehicle", "sequence-a:track-1"),),
        "train:1": (("other_vehicle", "sequence-a:track-1"),),
        "train:2": (("bicycle", "sequence-a:track-2"),),
        "train:3": (("bicycle", "sequence-b:track-3"),),
    }
    return CappedRareIdentitySampler(
        tokens,
        memberships,
        policy=rare_policy(),
        seed=7,
        source_role="train",
        manifest_sha256="a" * 64,
    )


def production_sized_sampler() -> CappedRareIdentitySampler:
    tokens = tuple(f"train:{index}" for index in range(4096))
    memberships = {
        "train:0": (("other_vehicle", "sequence-a:track-1"),),
        "train:1": (("other_vehicle", "sequence-a:track-1"),),
        "train:2": (("bicycle", "sequence-a:track-2"),),
        "train:3": (("bicycle", "sequence-b:track-3"),),
    }
    policy = RareIdentityPolicy(
        rare_classes=("bicycle", "other_vehicle"),
        rare_draw_fraction=0.25,
        max_frame_exposures_per_epoch=4,
        max_identity_exposure_multiplier=4,
    )
    return CappedRareIdentitySampler(
        tokens,
        memberships,
        policy=policy,
        seed=7,
        source_role="train",
        manifest_sha256="a" * 64,
    )


def test_photometric_rng_resume_is_exact_and_evaluation_is_off() -> None:
    policy = PhotometricPolicy(brightness=0.15, contrast=0.15, saturation=0.1)
    images = torch.linspace(-1, 1, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    targets = {"labels": torch.tensor([1, 2]), "depth": torch.tensor([[3.0], [4.0]])}
    targets_before = {name: value.clone() for name, value in targets.items()}
    uninterrupted = StatefulPhotometricAugmenter(policy, 7)
    uninterrupted.apply(images, training=True)
    state = uninterrupted.state_dict()
    expected = uninterrupted.apply(images, training=True)

    resumed = StatefulPhotometricAugmenter(policy, 7)
    resumed.load_state_dict(state)
    observed = resumed.apply(images, training=True)
    assert torch.equal(observed, expected)
    assert resumed.apply(images, training=False) is images
    assert all(torch.equal(targets[name], value) for name, value in targets_before.items())


def test_photometric_state_rejects_recipe_drift() -> None:
    first = StatefulPhotometricAugmenter(PhotometricPolicy(0.15, 0.15, 0.1), 7)
    second = StatefulPhotometricAugmenter(PhotometricPolicy(0.1, 0.15, 0.1), 7)
    with pytest.raises(ValueError, match="another recipe"):
        second.load_state_dict(first.state_dict())


def test_rare_sampler_preserves_budget_and_hard_caps() -> None:
    candidate = sampler()
    receipt = candidate.receipt
    assert receipt["total_exposures"] == 12
    assert sum(receipt["frame_exposures"].values()) == 12
    assert max(receipt["frame_exposures"].values()) <= 4
    for identity, exposure in receipt["identity_exposures"].items():
        assert exposure <= 4 * receipt["identity_support_frames"][identity]
    assert receipt["realized_rare_draws"] > 0
    assert receipt["source_role"] == "train"
    assert len(receipt["membership_sha256"]) == 64


def test_rare_sampler_resume_matches_future_token_order() -> None:
    uninterrupted = sampler()
    prefix = uninterrupted.next_batch(7)
    state = deepcopy(uninterrupted.state_dict())
    expected = [uninterrupted.next_batch(5) for _ in range(8)]
    resumed = sampler()
    resumed.load_state_dict(state)
    observed = [resumed.next_batch(5) for _ in range(8)]
    assert len(prefix) == 7
    assert observed == expected


def test_rare_sampler_rejects_selection_or_external_membership() -> None:
    with pytest.raises(ValueError, match="only from train"):
        CappedRareIdentitySampler(
            ("train:0",),
            {"train:0": (("other_vehicle", "sequence:track"),)},
            policy=rare_policy(),
            seed=7,
            source_role="selection",
            manifest_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="outside train"):
        CappedRareIdentitySampler(
            ("train:0",),
            {"selection:0": (("other_vehicle", "sequence:track"),)},
            policy=rare_policy(),
            seed=7,
            source_role="train",
            manifest_sha256="a" * 64,
        )


def test_rare_sampler_checkpoint_rejects_order_tampering() -> None:
    candidate = sampler()
    state = candidate.state_dict()
    state["order"] = list(reversed(state["order"]))
    with pytest.raises(ValueError, match="differs from its receipt"):
        sampler().load_state_dict(state)


def test_rare_sampler_persists_all_eight_completed_epoch_ledgers() -> None:
    candidate = production_sized_sampler()
    # The production arithmetic is 8,192 updates * batch four = eight D4096 epochs.
    for _ in range(8192):
        candidate.next_batch(4)
    state = candidate.state_dict()
    completed = state["completed_epoch_receipts"]
    assert candidate.epoch == 8
    assert len(completed) == 8
    assert [item["epoch"] for item in completed] == list(range(8))
    assert sum(item["total_exposures"] for item in completed) == 32768
    restored = production_sized_sampler()
    restored.load_state_dict(state)
    assert restored.completed_epoch_receipts == completed
    assert restored.next_batch(17) == candidate.next_batch(17)


def test_rare_sampler_recomputes_identity_ledgers_on_restore() -> None:
    candidate = sampler()
    candidate.next_batch(12)
    state = deepcopy(candidate.state_dict())
    identity = next(iter(state["completed_epoch_receipts"][0]["identity_exposures"]))
    state["completed_epoch_receipts"][0]["identity_exposures"][identity] = 999
    with pytest.raises(ValueError, match="identity exposure ledger"):
        sampler().load_state_dict(state)
