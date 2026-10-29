"""CPU contracts for opt-in strong RGB hashes and exact default V1 identity."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from operational.train40_runtime import rgb_hash_backend as backend_module
from operational.train40_runtime import rgb_raw_ring as frame_module
from operational.train40_runtime.feature_cache import ExactRGBFeatureCache, rgb_feature_key
from operational.train40_runtime.rgb_crop_cache import CROP_SHAPE, ExactRGBCropCache
from operational.train40_runtime.rgb_frozen_identity import bind_frozen_rgb_history
from operational.train40_runtime.rgb_hash_backend import VerifiedBlake3, load_verified_blake3
from operational.train40_runtime.rgb_raw_ring import (
    ResidentRGBFrameRing,
    RGBSourceIdentity,
    freeze_rgb_frame,
)

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "artifacts/train40_runtime_council_20261028/hash_probe_dependencies"


@pytest.fixture(scope="module")
def private_path():
    if not (PRIVATE / "download_receipt.json").is_file():
        pytest.skip("Optional pinned private BLAKE3 wheel is not installed")
    return PRIVATE


@pytest.fixture(scope="module")
def backend(private_path):
    before = sys.path[:]
    result = load_verified_blake3(private_path)
    assert sys.path == before
    return result


def token(index=1, *, backend=None, pixels=None):
    pixels = np.full((4, 5, 3), index, np.uint8) if pixels is None else pixels
    return freeze_rgb_frame(
        pixels,
        index,
        index * 100_000,
        RGBSourceIdentity("fixture", "a" * 64),
        hash_backend=backend,
    )


def feature_key(identity):
    return rgb_feature_key(
        "object",
        (0, 0, 5, 4),
        identity,
        model_sha256="b" * 64,
        runtime_sha256="c" * 64,
        batch_shape=(1, 2, 3, 128, 128),
        object_slot=0,
    )


def test_v1_default_and_none_keep_fixed_content_and_descriptor_digests():
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    args = (pixels, 7, 123, RGBSourceIdentity("fixture", "a" * 64))
    for result in (freeze_rgb_frame(*args), freeze_rgb_frame(*args, hash_backend=None)):
        assert result.token_version == 1 and result.content_algorithm == "sha256"
        assert result.content_sha256 == result.content_digest == result.sha256
        assert result.content_digest == (
            "0ddde28e40838ef6f9853e887f597d6adb5f40eb35d5763c52e1e64d8ba3bfff"
        )
        assert result.identity_sha256 == (
            "8296a81d4c045c47048b331661d744703d9ae7af7158c97cf206bfa7cda9e51d"
        )
    frames = {index: token(index) for index in (1, 2)}
    bound = bind_frozen_rgb_history(frames, ((1, 2),), query_us=200_000)
    legacy = (
        ("IMMUTABLE_OWNED_RGB_V1", "fixture", "a" * 64),
        ((1, 2),),
        [(i, frames[i].timestamp_us, frames[i].shape, frames[i].identity_sha256) for i in (1, 2)],
        (((0.1).hex(),),),
    )
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert bound.digest == expected


def test_default_v1_does_not_import_optional_package_in_fresh_process():
    code = """
import sys
import numpy as np
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
assert 'blake3' not in sys.modules
freeze_rgb_frame(np.zeros((1,1,3), np.uint8), 1, 1, RGBSourceIdentity('x', 'a'*64))
assert 'blake3' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_backend_receipt_thread_contract_and_independent_vector(backend, monkeypatch):
    calls = []
    record = backend_module._RECORDS[backend]

    def observed(payload, **kwargs):
        calls.append(kwargs)
        return record.constructor(payload, **kwargs)

    monkeypatch.setitem(backend_module._RECORDS, backend, replace(record, constructor=observed))
    assert backend.digest(b"") == (
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    assert calls == [{"max_threads": 1}]
    receipt = backend.receipt
    assert receipt["content_algorithm"] == "blake3-256" and receipt["digest_bytes"] == 32
    assert receipt["max_threads"] == 1 and receipt["installed_version"] == "1.0.11"
    assert set(receipt["protected_paths"]) == {str(path) for path in backend.protected_paths}
    assert all(path.is_file() for path in backend.protected_paths)
    receipt["content_algorithm"] = "tampered"
    assert backend.receipt["content_algorithm"] == "blake3-256"


def test_v2_hashes_bytes_once_and_sha_only_small_metadata(backend, monkeypatch):
    pixels = np.arange(1200, dtype=np.uint8).reshape(20, 20, 3)
    original_sha = hashlib.sha256
    sizes = []

    def metadata_only(data):
        assert len(data) < 1000, "V2 must never SHA the RAW payload"
        sizes.append(len(data))
        return original_sha(data)

    monkeypatch.setattr(frame_module.hashlib, "sha256", metadata_only)
    frames = {index: token(index, backend=backend, pixels=pixels) for index in (1, 2)}
    for result in frames.values():
        for name in ("sha256", "content_sha256"):
            with pytest.raises(ValueError, match="blake3-256.*content_digest"):
                getattr(result, name)
    bound = bind_frozen_rgb_history(frames, ((1, 2),), query_us=200_000)
    assert len(sizes) == 3 and bound.hashed_bytes == 0
    assert bound.source_binding == ("IMMUTABLE_OWNED_RGB_V2", "blake3-256", "fixture", "a" * 64)
    instance = ResidentRGBFrameRing(frame_shape=pixels.shape)
    first = instance.bind(list(frames.values()), 200_000)
    instance.mark_consumed(first)
    instance.close()
    assert len(sizes) == 3


def test_v2_deterministic_content_identity_and_immutable_ownership(backend):
    full = np.arange(120, dtype=np.uint8).reshape(4, 10, 3)
    pixels = full[:, ::2]
    expected = pixels.copy()
    first = token(backend=backend, pixels=pixels)
    second = token(backend=backend, pixels=expected)
    v1 = token(pixels=expected)
    assert first.content_algorithm == "blake3-256" and first.token_version == 2
    reference = (
        importlib.import_module("blake3")
        .blake3(expected.tobytes(), max_threads=1)
        .hexdigest(length=32)
    )
    assert first.content_digest == reference == second.content_digest
    assert first.identity_sha256 == second.identity_sha256 != v1.identity_sha256
    descriptor = (
        "IMMUTABLE_RGB_FRAME_V2",
        "blake3-256",
        "fixture",
        "a" * 64,
        1,
        100_000,
        (4, 5, 3),
        "uint8_HWC_RGB",
        reference,
    )
    assert (
        first.identity_sha256
        == hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()
    )
    pixels[:] = 0
    assert np.array_equal(first.array, expected) and not first.array.flags.writeable
    with pytest.raises(ValueError):
        first.array.setflags(write=True)
    with pytest.raises(AttributeError):
        object.__setattr__(first, "content_algorithm", "sha256")
    ref = weakref.ref(first)
    del first
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("change", ["pixels", "frame_id", "timestamp", "shape", "source"])
def test_v2_invalidates_every_content_and_provenance_change(backend, change):
    original = token(backend=backend)
    pixels, index, timestamp, source = original.array.copy(), 1, 100_000, original.source
    if change == "pixels":
        pixels[0, 0, 0] += 1
    elif change == "frame_id":
        index = 2
    elif change == "timestamp":
        timestamp += 1
    elif change == "shape":
        pixels = pixels.reshape(5, 4, 3)
    else:
        source = RGBSourceIdentity("other", "a" * 64)
    changed = freeze_rgb_frame(pixels, index, timestamp, source, hash_backend=backend)
    assert original.identity_sha256 != changed.identity_sha256


def test_mixed_profiles_fail_before_ring_mutation_and_metadata_hash(backend, monkeypatch):
    frames = [token(1), token(2, backend=backend)]
    instance = ResidentRGBFrameRing(frame_shape=(4, 5, 3))
    before = instance.stats
    monkeypatch.setattr(frame_module.hashlib, "sha256", lambda *_: pytest.fail("mixed hash"))
    with pytest.raises(ValueError, match="hash profiles"):
        bind_frozen_rgb_history(dict(zip((1, 2), frames, strict=True)), ((1, 2),), query_us=200_000)
    with pytest.raises(ValueError, match="hash profiles"):
        instance.bind(frames, 200_000)
    assert instance.stats == before
    instance.close()


def test_ring_profile_transition_resets_and_repeat_hits(backend):
    instance = ResidentRGBFrameRing(frame_shape=(4, 5, 3))
    for current, reset, hits in (
        (token(), False, 0),
        (token(backend=backend), True, 0),
        (token(backend=backend), False, 1),
        (token(), True, 0),
    ):
        bound = instance.bind([current], 100_000)
        assert bound.receipt["hash_profile_reset"] == reset
        assert bound.receipt["source_reset"] is False
        assert bound.receipt["hits"] == hits
        assert bound.receipt["content_algorithm"] == current.content_algorithm
        assert bound.receipt["frame_identity_algorithm"] == "sha256"
        assert np.array_equal(bound.tensor[bound.slots[0]].numpy(), current.array)
        instance.mark_consumed(bound)
    assert instance.stats["hash_profile_resets"] == 2
    instance.reset()
    bound = instance.bind([token(backend=backend)], 100_000)
    assert not bound.receipt["hash_profile_reset"]
    instance.mark_consumed(bound)
    instance.close()


def test_both_caches_miss_across_v1_v2_with_identical_pixels(backend):
    histories = [
        {index: token(index, backend=option) for index in (1, 2)} for option in (None, backend)
    ]
    identities = [
        bind_frozen_rgb_history(frames, ((1, 2),), query_us=200_000) for frames in histories
    ]
    keys = [feature_key(identity) for identity in identities]
    assert keys[0].digest != keys[1].digest
    feature_cache = ExactRGBFeatureCache()
    assert feature_cache.store_many(
        [keys[0]], torch.ones(1, 1, 17), torch.ones(1, 1, 3), guard_passed=True
    )
    assert feature_cache.lookup_many([keys[1]]) is None
    crop_cache = ExactRGBCropCache()
    roi, pixels = (0, 0, 5, 4), np.zeros(CROP_SHAPE, np.float32)
    assert crop_cache.put("object", histories[0][1], roi, pixels, query_us=200_000)
    assert crop_cache.get("object", histories[1][1], roi, query_us=200_000) is None
    assert crop_cache.put("object", histories[1][1], roi, pixels, query_us=200_000)
    cached = crop_cache.get("object", histories[1][1], roi, query_us=200_000)
    assert cached is not None and np.array_equal(cached, pixels)
    crop_cache.close()


def test_missing_and_forged_backends_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_verified_blake3(tmp_path)
    with pytest.raises(TypeError, match="load_verified_blake3"):
        VerifiedBlake3()
    with pytest.raises(ValueError, match="forged"):
        token(backend=object.__new__(VerifiedBlake3))
    with pytest.raises(ValueError, match="factory-verified"):
        token(backend=lambda payload: "0" * 64)


@pytest.mark.parametrize("change", ["receipt", "wheel", "metadata", "installed"])
def test_dependency_tampering_is_rejected_before_import(private_path, tmp_path, change):
    target = tmp_path / "private"
    shutil.copytree(private_path, target)
    if change == "receipt":
        path = target / "download_receipt.json"
        data = json.loads(path.read_text())
        data["version"] = "other"
        path.write_text(json.dumps(data))
    else:
        path = {
            "wheel": target / "wheels" / "blake3-1.0.11-cp311-cp311-win_amd64.whl",
            "metadata": target / "pypi_release_metadata.json",
            "installed": target / "site" / "blake3" / "__init__.py",
        }[change]
        path.write_bytes(path.read_bytes() + b"tampered")
    before = sys.path[:]
    with pytest.raises(ValueError, match="pinned|altered"):
        load_verified_blake3(target)
    assert sys.path == before


@pytest.mark.parametrize("name", ["blake3", "blake3.blake3"])
def test_foreign_preloaded_module_or_extension_rejected(private_path, monkeypatch, name):
    foreign = ModuleType(name)
    foreign.__file__ = "foreign.py"
    monkeypatch.setitem(sys.modules, name, foreign)
    before = sys.path[:]
    with pytest.raises(ValueError, match="unverified origin"):
        load_verified_blake3(private_path)
    assert sys.path == before
