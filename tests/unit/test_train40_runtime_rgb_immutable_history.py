"""Immutable pixel ownership and exact cache identity without a CUDA runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from operational.train40_runtime import multiobject_cache as cache_module
from operational.train40_runtime.multiobject import (
    MultiObjectTrain40Service,
    ObjectRegion,
    RawRGBHistory,
    rgb_metadata,
)
from operational.train40_runtime.multiobject_cache import RGBBatchCache
from operational.train40_runtime.rgb_immutable_history import FrozenRGBHistory
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
from scripts.benchmark_train40_multiobject import reference_rgb


def tokens():
    origin = RGBSourceIdentity("sensor", "a" * 64)
    return {
        index: freeze_rgb_frame(
            np.full((4, 5, 3), index, np.uint8), index, (index + 1) * 100_000, origin
        )
        for index in range(7)
    }


def history(query=700_000, frames=None):
    return FrozenRGBHistory(
        query,
        tokens() if frames is None else frames,
        tuple((index, index + 1, index + 2) for index in range(5)),
    )


def test_immutable_history_snapshots_tokens_and_regenerates_array_metadata():
    original = tokens()
    packet = history(frames=original)
    assert isinstance(packet, RawRGBHistory)
    original.clear()
    assert len(packet.frames) == len(packet.tokens) == 7
    first = packet.frames[0]
    first.shape = (5, 4, 3)
    assert packet.frames[0].shape == (4, 5, 3)
    assert np.shares_memory(first, packet.frames[0])
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(TypeError):
        packet.tokens[0] = packet.tokens[1]
    with pytest.raises(TypeError):
        packet.timestamps_us[0] = 123
    with pytest.raises(FrozenInstanceError):
        packet.tokens = {}
    with pytest.raises(FrozenInstanceError):
        packet.frames = {}


@pytest.mark.parametrize("invalid", ["extra", "missing", "clock", "source", "unavailable"])
def test_inconsistent_immutable_history_is_rejected(invalid):
    frames = tokens()
    groups = tuple((index, index + 1, index + 2) for index in range(5))
    if invalid == "extra":
        frames[10] = frames[0]
    elif invalid == "missing":
        del frames[0]
    elif invalid == "source":
        frames[0] = freeze_rgb_frame(
            frames[0].array, 0, 100_000, RGBSourceIdentity("other", "b" * 64)
        )
    with pytest.raises(ValueError):
        FrozenRGBHistory(
            699_999 if invalid == "clock" else 700_000,
            frames,
            groups,
            "missing" if invalid == "unavailable" else "",
        )


def test_unavailable_history_has_no_borrowed_frame_state():
    packet = FrozenRGBHistory(700_000, {}, (), "RGB_DATASET_MISSING")
    assert not packet.tokens and not packet.frames and not packet.triplets


def test_immutable_history_cannot_override_token_pixel_contract():
    with pytest.raises(TypeError, match="cannot override"):
        type("OtherHistory", (FrozenRGBHistory,), {})


@pytest.mark.parametrize("count", [1, 2])
def test_actual_pil_crop_inputs_match_independent_mutable_source_oracle(count):
    origin = RGBSourceIdentity("sensor", "a" * 64)
    rng = np.random.default_rng(7)
    raw = {index: rng.integers(0, 256, (31, 47, 3), dtype=np.uint8) for index in range(7)}
    frozen = {
        index: freeze_rgb_frame(frame, index, (index + 1) * 100_000, origin)
        for index, frame in raw.items()
    }
    immutable = history(frames=frozen)
    control = RawRGBHistory(700_000, raw, immutable.timestamps_us, immutable.triplets)
    regions = [
        ObjectRegion(str(index), (0, 0, 31, 47), (-5.2 + index, 1.3, 32.7 + index, 39.2))
        for index in range(count)
    ]
    service = MultiObjectTrain40Service.__new__(MultiObjectTrain40Service)
    service.shared_rgb_images = False
    with ThreadPoolExecutor(max_workers=2) as workers:
        service._crops = workers
        actual = service._rgb(regions, immutable)
    for region, outputs in zip(regions, actual, strict=True):
        expected = reference_rgb(control, region)
        for (pixels, metadata), (expected_pixels, expected_metadata) in zip(
            outputs, expected, strict=True
        ):
            np.testing.assert_array_equal(pixels, expected_pixels)
            assert metadata == expected_metadata


@pytest.mark.parametrize("count", [1, 2])
def test_cache_hits_later_causal_query_without_scanning_raw_or_caching_context(monkeypatch, count):
    frames = tokens()
    first, later = history(frames=frames), history(720_000, frames)
    regions = [ObjectRegion(str(index), (0, 0, 4, 5), (0, 0, 4, 5)) for index in range(count)]
    cache = RGBBatchCache(
        model_sha256="b" * 64,
        runtime_sha256="c" * 64,
        rgb_object_chunk_size=1,
        max_entries=count,
    )
    monkeypatch.setattr(
        cache_module, "bind_rgb_history", lambda *args, **kwargs: pytest.fail("RAW rehash")
    )
    ticket = cache.begin(regions, first)
    assert ticket.cached_features is None and ticket.diagnostics["hashed_bytes"] == 0
    assert all(key.batch_shape == (5, 3, 3, 128, 128) for key in ticket.keys)
    assert all(key.object_slot == 0 for key in ticket.keys)
    raw, experts = torch.arange(count * 85).reshape(count * 5, 17).float(), torch.ones(count * 5, 3)
    assert cache.store(ticket, raw, experts, guard_passed=True)
    hit = cache.begin(regions, later)
    assert hit.cached_features is not None
    assert torch.equal(torch.cat([item.raw for item in hit.cached_features]), raw)
    assert hit.identity.digest == ticket.identity.digest
    assert rgb_metadata(regions, first)[0][-1]["available_us"] == 700_000
    assert rgb_metadata(regions, later)[0][-1]["available_us"] == 720_000


def test_arriving_new_pixels_invalidate_immutable_cache_identity():
    frames = tokens()
    regions = [ObjectRegion("car", (0, 0, 4, 5), (0, 0, 4, 5))]
    cache = RGBBatchCache(model_sha256="b" * 64, runtime_sha256="c" * 64)
    first = cache.begin(regions, history(frames=frames))
    assert cache.store(first, torch.ones(5, 17), torch.ones(5, 3), guard_passed=True)
    token = frames[6]
    frames[6] = freeze_rgb_frame(np.full((4, 5, 3), 99, np.uint8), 6, 700_000, token.source)
    changed = cache.begin(regions, history(720_000, frames))
    assert changed.cached_features is None and changed.identity.digest != first.identity.digest
