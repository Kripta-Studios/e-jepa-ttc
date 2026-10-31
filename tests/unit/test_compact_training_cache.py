from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler
from e_jepa_ttc.detection.compact_training_cache import (
    CompactTrainingCache,
    decode_image,
    encode_image,
    normalization_table,
    peek_permutation_batch,
)
from e_jepa_ttc.detection.monodgp_eap_adapter import PreparedSample


def prepared():
    codes = torch.arange(3 * 64 * 64).remainder(256).to(torch.uint8).reshape(3, 64, 64)
    return PreparedSample(
        decode_image(codes, normalization_table()),
        torch.zeros(3, 4),
        torch.tensor([64, 64]),
        {
            "obj_region": torch.arange(64 * 64).remainder(2).reshape(64, 64).float(),
            "labels": torch.tensor([1]),
            "velocity": torch.tensor([[2.0, -3.0]]),
        },
        [],
        {"sample_token": "a"},
    )


def test_every_rgb_value_round_trips_exactly():
    codes = torch.arange(256).to(torch.uint8).repeat(3, 1).reshape(3, 16, 16)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (codes.float() / 255 - mean) / std
    encoded = encode_image(image, normalization_table())
    assert torch.equal(encoded, codes)
    assert torch.equal(decode_image(encoded, normalization_table()), image)


def test_nonrepresentable_image_is_rejected():
    image = prepared().image
    image[0, 0, 0] += 0.0001
    with pytest.raises(ValueError, match="would change"):
        encode_image(image, normalization_table())


class Dataset:
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.reads += 1
        return SimpleNamespace(sample_token="a", member_sha256="expected")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_and_first_read_preserve_all_training_tensors(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dataset = Dataset()
    expected = prepared()
    cache = CompactTrainingCache(
        dataset,
        SimpleNamespace(model=SimpleNamespace(width=64, height=64)),
        {"a": "expected"},
        prepare=lambda raw: expected,
        device=device,
        gpu_image_budget=3 * 64 * 64,
        cpu_image_budget=3 * 64 * 64,
    )
    cache.prefetch([0])
    for _ in range(2):
        actual, identity = cache.get(0)
        assert identity.sample_token == "a"
        assert not hasattr(identity, "image")
        assert torch.equal(actual.image.cpu(), expected.image)
        assert torch.equal(actual.calibration, expected.calibration)
        assert actual.report == expected.report
        for key, value in actual.target.items():
            assert torch.equal(value.cpu(), expected.target[key])
    assert dataset.reads == 1
    assert cache.statistics()["raw_images_retained"] == 0
    cache.close()


def test_memory_budget_is_checked_before_population_allocation():
    with pytest.raises(ValueError, match="CPU image budget"):
        CompactTrainingCache(
            Dataset(),
            SimpleNamespace(model=SimpleNamespace(width=64, height=64)),
            {"a": "expected"},
            prepare=lambda raw: prepared(),
            device="cpu",
            cpu_image_budget=1,
        )


@pytest.mark.parametrize("corruption", ["member", "region"])
def test_corrupt_media_or_nonbinary_target_cannot_enter_cache(corruption):
    value = prepared()
    if corruption == "region":
        value.target["obj_region"][0, 0] = 0.5
    cache = CompactTrainingCache(
        Dataset(),
        SimpleNamespace(model=SimpleNamespace(width=64, height=64)),
        {"a": "wrong" if corruption == "member" else "expected"},
        prepare=lambda raw: value,
        device="cpu",
    )
    with pytest.raises(ValueError):
        cache.get(0)
    cache.close()


def test_prefetch_peek_preserves_sampler_and_rng_across_epochs():
    sampler = ResumablePermutationSampler(7, 23)
    rng = torch.get_rng_state().clone()
    for _ in range(30):
        before = sampler.state_dict()
        expected = peek_permutation_batch(sampler, 4)
        after = sampler.state_dict()
        for key in ("size", "seed", "epoch", "position", "order"):
            assert before[key] == after[key]
        assert torch.equal(before["generator_state"], after["generator_state"])
        assert expected == sampler.next_batch(4)
    assert torch.equal(rng, torch.get_rng_state())


def test_prefetch_queue_is_bounded_even_when_completed_rows_are_not_consumed():
    class Many(Dataset):
        def __len__(self):
            return 10

    cache = CompactTrainingCache(
        Many(),
        SimpleNamespace(model=SimpleNamespace(width=64, height=64)),
        {"a": "expected"},
        prepare=lambda raw: prepared(),
        device="cpu",
    )
    try:
        cache.prefetch(list(range(8)))
        with pytest.raises(ValueError, match="queue exceeded"):
            cache.prefetch([8])
    finally:
        cache.close()
