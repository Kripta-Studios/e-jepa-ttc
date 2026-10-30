"""Residency must preserve exact values, bounds, identity checks and RNG state."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache


@dataclass
class Prepared:
    image: torch.Tensor
    target: dict


def fixture_cache(**kwargs):
    dataset = [SimpleNamespace(sample_token=str(i), member_sha256=f"sha{i}") for i in range(4)]
    calls = []

    def prepare(raw, label_free):
        calls.append((raw.sample_token, label_free))
        return Prepared(torch.full((2, 3), float(raw.sample_token)), {"label_free": label_free})

    cache = ResidentInputCache(
        dataset, None, kwargs.pop("member_sha256", {}), prepare=prepare, device="cpu", **kwargs
    )
    return cache, calls


def test_exact_reuse_retains_targets_and_rng():
    cache, calls = fixture_cache()
    rng = torch.get_rng_state().clone()
    first, raw = cache.get(2)
    again, same_raw = cache.get(2)
    assert first is again and raw is same_raw
    assert torch.equal(first.image, torch.full((2, 3), 2.0))
    assert first.target == {"label_free": False}
    assert torch.equal(rng, torch.get_rng_state())
    assert len(calls) == 1 and cache.hits == 1


def test_capacity_and_byte_budget_evict_lru():
    cache, _ = fixture_cache(train_capacity=3, maximum_image_bytes=48)
    for i in (0, 1, 0, 2):
        cache.get(i)
    assert list(cache.cache) == [0, 2]
    assert cache.image_bytes == cache.peak_image_bytes == 48
    cache.get(1)
    assert list(cache.cache) == [2, 1]


def test_changed_member_fails_before_preparation():
    cache, calls = fixture_cache(member_sha256={"0": "wrong"})
    with pytest.raises(ValueError, match="RGB member bytes changed"):
        cache.get(0)
    assert not calls and not cache.cache and cache.image_bytes == 0


def test_evaluation_keeps_label_free_cpu_and_small_capacity():
    cache, calls = fixture_cache(label_free_model_input=True, evaluation_capacity=1)
    for i in (0, 1):
        prepared, _ = cache.get(i)
        assert prepared.image.device.type == "cpu"
        assert prepared.target == {"label_free": True}
    assert len(cache.cache) == 1 and all(flag for _, flag in calls)


def test_oversized_sample_does_not_change_committed_cache():
    cache, _ = fixture_cache(maximum_image_bytes=16)
    with pytest.raises(ValueError, match="exceeds"):
        cache.get(0)
    assert not cache.cache and not cache.member_sha256
