"""Exact DINO parity and memory bounds without model or GPU initialization."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from operational.rgb_port_rgb_loader.loader import installed_dino_cache


class Source:
    """Original-reader equivalent, independent of the optimized implementation."""

    def __init__(self, root: Path, count: int = 9) -> None:
        self._dino_values = self._dino_valid = None
        self._dino_shards: list[dict[str, Any]] = []
        self._dino_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.reads = 0
        rows = []
        for shard in range(count):
            start = 2 * shard
            path = root / f"shard_{shard}.npz"
            ordinals = np.arange(start, start + 2, dtype=np.int64)
            tokens = np.asarray([f"token-{ordinal}" for ordinal in ordinals])
            np.savez_compressed(
                path,
                ordinals=ordinals,
                tokens=tokens,
                relation_targets=ordinals[:, None, None].astype(np.float16),
                relation_valid=np.asarray([[[True]], [[False]]]),
            )
            self._dino_shards.append({"start": start, "stop": start + 2, "path": str(path)})
            rows.extend(
                {"source_ordinal": int(ordinal), "sample_token": str(token)}
                for ordinal, token in zip(ordinals, tokens, strict=True)
            )
        self.dataset = SimpleNamespace(rows=rows)

    def _dino_for_indices(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if self._dino_values is not None and self._dino_valid is not None:
            return self._dino_values[indices], self._dino_valid[indices]
        values, valid = [], []
        for index in indices:
            row = self.dataset.rows[index]
            ordinal = int(row["source_ordinal"])
            item = next(
                (
                    shard
                    for shard in self._dino_shards
                    if int(shard["start"]) <= ordinal < int(shard["stop"])
                ),
                None,
            )
            if item is None:
                raise ValueError(f"no generic-DINO shard covers source ordinal {ordinal}")
            key = str(item["path"])
            stored = self._dino_cache.get(key)
            if stored is None:
                with np.load(key, allow_pickle=False) as payload:
                    stored = {name: payload[name] for name in payload.files}
                self.reads += 1
                self._dino_cache[key] = stored
                while len(self._dino_cache) > 4:
                    self._dino_cache.popitem(last=False)
            else:
                self._dino_cache.move_to_end(key)
            local = ordinal - int(item["start"])
            if int(stored["ordinals"][local]) != ordinal or str(stored["tokens"][local]) != str(
                row["sample_token"]
            ):
                raise ValueError("generic-DINO shard row identity mismatch")
            values.append(stored["relation_targets"][local])
            valid.append(stored["relation_valid"][local])
        return np.stack(values), np.stack(valid)


def test_exact_values_order_dtypes_and_fewer_reads(tmp_path: Path) -> None:
    source = Source(tmp_path)
    ids = list(range(0, 18, 2)) + list(range(1, 18, 2)) + [0, 5, 0]
    expected = source._dino_for_indices(ids)
    baseline_reads = source.reads
    original_method = source._dino_for_indices
    with installed_dino_cache(source) as cache:
        actual = source._dino_for_indices(ids)
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(left, right, strict=True)
        assert cache.reads == 9 < baseline_reads
        assert cache.hits == len(ids) - 9
        assert cache.snapshot()["retained_shards"] == 9
    assert source._dino_for_indices == original_method
    assert cache.retained_bytes == 0
    np.testing.assert_array_equal(actual[0], expected[0], strict=True)


@pytest.mark.parametrize("max_shards,max_bytes", [(2, 100000), (32, 250), (32, 1)])
def test_both_cache_bounds_preserve_output(tmp_path: Path, max_shards: int, max_bytes: int) -> None:
    source = Source(tmp_path)
    ids = list(range(18)) * 2
    expected = source._dino_for_indices(ids)
    with installed_dino_cache(source, max_shards=max_shards, max_bytes=max_bytes) as cache:
        actual = source._dino_for_indices(ids)
        assert len(cache.cache) <= max_shards
        assert cache.retained_bytes <= max_bytes
        assert cache.peak_retained_bytes <= max_bytes
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(left, right, strict=True)


@pytest.mark.parametrize("kind", ["token", "ordinal", "missing", "missing_file"])
def test_validation_and_io_failures_remain_visible(tmp_path: Path, kind: str) -> None:
    source = Source(tmp_path, count=1)
    if kind == "token":
        source.dataset.rows[0]["sample_token"] = "wrong"
    elif kind == "ordinal":
        path = Path(source._dino_shards[0]["path"])
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        arrays["ordinals"][0] = 42
        np.savez_compressed(path, **arrays)
    elif kind == "missing":
        source.dataset.rows[0]["source_ordinal"] = 42
    else:
        Path(source._dino_shards[0]["path"]).unlink()
    with installed_dino_cache(source), pytest.raises((ValueError, FileNotFoundError)):
        source._dino_for_indices([0])


def test_overlapping_metadata_preserves_first_match(tmp_path: Path) -> None:
    source = Source(tmp_path, count=2)
    # An overlapping entry appearing last must not override the original first.
    source._dino_shards.append({"start": 0, "stop": 4, "path": "missing.npz"})
    expected = source._dino_for_indices([0, 3])
    with installed_dino_cache(source):
        actual = source._dino_for_indices([0, 3])
    np.testing.assert_array_equal(expected[0], actual[0], strict=True)


def test_instance_override_restored_on_exception(tmp_path: Path) -> None:
    source = Source(tmp_path, count=1)
    original = source._dino_for_indices
    source._dino_for_indices = original  # type: ignore[method-assign]
    with pytest.raises(RuntimeError), installed_dino_cache(source):
        raise RuntimeError("caller failed")
    assert vars(source)["_dino_for_indices"] == original


def test_monolithic_source_delegates_without_shard_cache(tmp_path: Path) -> None:
    source = Source(tmp_path, count=1)
    source._dino_values = np.asarray([[1.0], [2.0]], dtype=np.float32)  # type: ignore[assignment]
    source._dino_valid = np.asarray([[True], [False]])  # type: ignore[assignment]
    expected = source._dino_for_indices([1, 0])
    with installed_dino_cache(source) as cache:
        actual = source._dino_for_indices([1, 0])
        assert cache.reads == cache.hits == 0
    np.testing.assert_array_equal(expected[0], actual[0], strict=True)


@pytest.mark.parametrize("max_shards,max_bytes", [(0, 10), (2, 0)])
def test_invalid_bounds_fail_before_replacing_source(
    tmp_path: Path, max_shards: int, max_bytes: int
) -> None:
    source = Source(tmp_path, count=1)
    original = source._dino_for_indices
    with (
        pytest.raises(ValueError),
        installed_dino_cache(source, max_shards=max_shards, max_bytes=max_bytes),
    ):
        pass
    assert source._dino_for_indices == original
