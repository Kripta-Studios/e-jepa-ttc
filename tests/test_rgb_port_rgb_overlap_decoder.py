"""Exact prepared-RGB reads, identity validation and bounded decoder lifetimes."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from e_jepa_ttc.rgb_port.data import PreparedRGBCache
from operational.rgb_port_rgb_overlap.decoder import installed_decoder


class Dataset:
    def __init__(self, cache: PreparedRGBCache, rows: list[dict[str, Any]]) -> None:
        self.prepared_cache = cache
        self.rows = rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.prepared_cache.get(index, self.rows[index]["sample_token"])


class Source:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def batch(self, indices: list[int], modality: str) -> list[dict[str, Any]]:
        assert modality == "rgb"
        return [self.dataset[index] for index in indices]


def source_fixture(tmp_path: Path) -> Source:
    cache = object.__new__(PreparedRGBCache)
    cache._lookup = {}
    cache._loaded = OrderedDict()
    cache._loaded_bytes = 0
    cache._max_loaded_bytes = 256 * 1024**2
    rows = []
    for number in range(6):
        path = tmp_path / f"rgb_{number}.npz"
        indices = np.arange(number * 2, number * 2 + 2)
        tokens = np.asarray([f"token-{index}" for index in indices])
        np.savez_compressed(
            path,
            indices=indices,
            tokens=tokens,
            rgb_uint8=np.full((2, 3, 3, 4, 4), number, dtype=np.uint8),
            delta_t_s=np.asarray([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        )
        shard = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tokens": tokens.tolist(),
        }
        for local, index in enumerate(indices):
            cache._lookup[int(index)] = (shard, local)
            rows.append({"sample_token": str(tokens[local])})
    return Source(Dataset(cache, rows))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_exact_all_fields_and_restore_with_worker_bounds(tmp_path: Path, workers: int) -> None:
    source = source_fixture(tmp_path)
    ids = [6, 0, 5, 9, 2, 10, 7, 1, 4, 8, 3, 11]
    expected = source.batch(ids, "rgb")
    original_cache = source.dataset.prepared_cache
    original_cache._loaded.clear()
    original_cache._loaded_bytes = 0
    original_method = source.batch
    with installed_decoder(source, workers=workers) as cache:
        actual = source.batch(ids, "rgb")
        stats = cache.snapshot()
        assert stats["parallel_reads"] == 6
        assert stats["peak_pending_reads"] <= workers
        assert stats["peak_pending_bytes"] <= stats["max_pending_bytes"]
        assert cache.pending_bytes == 0
    assert source.dataset.prepared_cache is original_cache
    assert source.batch == original_method
    for left, right in zip(expected, actual, strict=True):
        assert left.keys() == right.keys()
        for field in left:
            np.testing.assert_array_equal(left[field], right[field], strict=True)


def test_small_reservation_falls_back_without_parallel_allocation(tmp_path: Path) -> None:
    source = source_fixture(tmp_path)
    with installed_decoder(source, workers=4, max_pending_bytes=1) as cache:
        actual = source.batch([0, 2, 4], "rgb")
        assert len(actual) == 3
        assert cache.reads == cache.peak_pending_reads == cache.peak_pending_bytes == 0
        assert cache.fallback_reads == 3


@pytest.mark.parametrize("mutation", ["hash", "tokens", "row_token"])
def test_canonical_identity_checks_are_not_skipped(tmp_path: Path, mutation: str) -> None:
    source = source_fixture(tmp_path)
    shard = source.dataset.prepared_cache._lookup[0][0]
    if mutation == "hash":
        shard["sha256"] = "0" * 64
    elif mutation == "tokens":
        shard["tokens"] = ["wrong", "wrong"]
    else:
        source.dataset.rows[0]["sample_token"] = "wrong"
    with installed_decoder(source), pytest.raises(ValueError, match="hash|token"):
        source.batch([0, 2], "rgb")


def test_failed_future_is_not_cached_across_original_retry(tmp_path: Path) -> None:
    source = source_fixture(tmp_path)
    original_batch = source.batch

    def retry(indices: list[int], modality: str) -> list[dict[str, Any]]:
        try:
            return original_batch(indices, modality)
        except FileNotFoundError:
            return original_batch(indices, modality)

    source.batch = retry  # type: ignore[method-assign]
    with installed_decoder(source) as cache:
        original_read = cache._read
        attempts = 0

        def flaky(index: int, token: str, key: str) -> Any:
            nonlocal attempts
            attempts += 1
            if index == 0:
                raise FileNotFoundError("transient disk disappearance")
            return original_read(index, token, key)

        cache._read = flaky  # type: ignore[method-assign]
        actual = source.batch([0, 2], "rgb")
        assert len(actual) == 2
        assert attempts == 2
        assert cache.fallback_reads == 1
    assert source.batch is retry


def test_eviction_preserves_lru_and_arrays_already_returned(tmp_path: Path) -> None:
    source = source_fixture(tmp_path)
    original = source.dataset.prepared_cache
    original._max_loaded_bytes = 1  # canonical rule retains at least one shard
    ids = [0, 2, 4, 0, 3, 5]
    expected = source.batch(ids, "rgb")
    expected_keys = list(original._loaded)
    original._loaded.clear()
    original._loaded_bytes = 0
    with installed_decoder(source) as cache:
        actual = source.batch(ids, "rgb")
        assert list(original._loaded) == expected_keys
        assert cache.fallback_reads > 0
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left["rgb"], right["rgb"], strict=True)


def test_restoration_and_worker_join_on_error(tmp_path: Path) -> None:
    source = source_fixture(tmp_path)
    original = source.dataset.prepared_cache
    with pytest.raises(IndexError), installed_decoder(source) as cache:
        source.batch([0, 99, 2], "rgb")
    assert source.dataset.prepared_cache is original
    assert cache.closed
    assert not cache.pending
    assert "batch" not in vars(source)


@pytest.mark.parametrize("workers,budget", [(0, 10), (1, 0)])
def test_invalid_bounds_do_not_replace_source(tmp_path: Path, workers: int, budget: int) -> None:
    source = source_fixture(tmp_path)
    original = source.dataset.prepared_cache
    with (
        pytest.raises(ValueError),
        installed_decoder(source, workers=workers, max_pending_bytes=budget),
    ):
        pass
    assert source.dataset.prepared_cache is original
