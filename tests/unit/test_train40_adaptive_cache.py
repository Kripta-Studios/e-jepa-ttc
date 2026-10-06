"""CPU-only parity and pressure tests for the TRAIN40 adaptive shard cache."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from operational.train40_system.adaptive_cache import (
    POLICY,
    AdaptiveSealedInputs,
    CachePolicy,
    MemorySnapshot,
)
from operational.train40_system.engine_overlap import SealedInputs


def _fixture(output: Path, *, shards: int = 8) -> int:
    cache = output / "input"
    teacher = output / "teacher"
    cache.mkdir(parents=True)
    teacher.mkdir()
    rows: list[dict[str, str]] = []
    input_entries = []
    teacher_entries = []
    value_bytes = 0
    for number in range(shards):
        start = number * 32
        ordinals = np.arange(start, start + 32, dtype=np.int64)
        tokens = np.asarray([f"token-{index}" for index in ordinals])
        rng = np.random.default_rng(100 + number)
        values = {
            "events": rng.normal(size=(32, 3, 12, 8, 8)).astype(np.float32),
            "delta": np.full(32, 0.1, dtype=np.float32),
            "motion": rng.normal(size=(32, 18)).astype(np.float32),
            "visible_heights": np.full((32, 2), [8.0, 9.0], dtype=np.float32),
            "target_ttc": np.full(32, -2.0, dtype=np.float32),
            "boxes": np.tile(
                np.asarray([[0, 0, 4, 4], [1, 1, 5, 5], [2, 2, 6, 6]], np.float32),
                (32, 1, 1),
            ),
            "square": np.tile(np.asarray([0, 0, 8, 8], np.float32), (32, 1)),
            "ordinals": ordinals,
            "tokens": tokens,
        }
        relations = rng.normal(size=(32, 2, 6, 2, 2)).astype(np.float16)
        valid = np.ones_like(relations, dtype=bool)
        input_path = cache / f"shard_{number:05d}.npz"
        teacher_path = teacher / input_path.name
        np.savez(input_path, **values)
        np.savez(
            teacher_path,
            ordinals=ordinals,
            tokens=tokens,
            relation_targets=relations,
            relation_valid=valid,
        )
        value_bytes = sum(array.nbytes for array in values.values())
        value_bytes += relations.nbytes + valid.nbytes
        for path, entries in ((input_path, input_entries), (teacher_path, teacher_entries)):
            stat = path.stat()
            entries.append(
                {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
        rows.extend(
            {
                "sequence_id": "sequence",
                "sample_token": str(token),
                "track_id": f"track-{index}",
            }
            for index, token in zip(ordinals, tokens, strict=True)
        )
    pd.DataFrame(rows).to_parquet(output / "TRAIN40_ROWS.parquet", index=False)
    (output / "PREPARE_FREEZE.json").write_text(
        json.dumps({"cache_root": str(cache)}), encoding="utf-8"
    )
    for kind, directory, entries in (
        ("INPUT", cache, input_entries),
        ("TEACHER", teacher, teacher_entries),
    ):
        (output / f"{kind}_MANIFEST.json").write_text(
            json.dumps({"directory": str(directory), "ordered_shards": entries}),
            encoding="utf-8",
        )
    return value_bytes


def _policy(*, absolute: int = 10**9) -> CachePolicy:
    return CachePolicy(
        absolute_cache_bytes=absolute,
        host_reserve_bytes=300,
        host_trim_target_bytes=325,
        tree_rss_soft_cap_bytes=10_000,
    )


class Probe:
    def __init__(self, available: int = 390, rss: int = 1_000) -> None:
        self.available = available
        self.rss = rss
        self.calls = 0
        self.lock = Lock()

    def __call__(self) -> MemorySnapshot:
        with self.lock:
            self.calls += 1
            return MemorySnapshot(self.available, self.rss)


def _assert_batches_equal(left: Any, right: Any) -> None:
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), name
        else:
            assert a == b, name


def test_exact_shard_and_batch_parity_with_frozen_loader(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=2)
    original = SealedInputs(tmp_path)
    adaptive = AdaptiveSealedInputs(tmp_path, memory_probe=Probe())
    try:
        expected, actual = original.shard(0), adaptive.shard(0)
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        ids = [0, 7, 31, 32, 47, 63]
        _assert_batches_equal(adaptive.batch(ids), original.batch(ids))
        json.dumps(adaptive.snapshot())
    finally:
        original.close()
        adaptive.close()


def test_near_old_four_gib_threshold_retains_stable_cache(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=4)
    probe = Probe(available=390)
    cache = AdaptiveSealedInputs(tmp_path, policy=_policy(), memory_probe=probe)
    try:
        for number in range(4):
            cache.shard(number)
        assert len(cache.cache) == 4
        assert cache.evictions == cache.admission_skips == 0
        assert probe.calls == 4
    finally:
        cache.close()


def test_constant_pressure_is_credited_once_and_skips_growth(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=16)
    probe = Probe(available=390)
    cache = AdaptiveSealedInputs(
        tmp_path,
        policy=_policy(),
        memory_probe=probe,
    )
    try:
        for number in range(4):
            cache.shard(number)
        probe.available = 290  # One 35-byte trim deficit for the pressure episode.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(cache.shard, range(4, 8)))
        assert cache.evictions == 1
        assert len(cache.cache) == 3
        assert cache.admission_skips == 4
        assert cache.pressure_episodes == 1
        assert cache.snapshot()["projected_release_credit_bytes"] == 35
        # A continuing pressure observation cannot expire and repeatedly drain the LRU.
        for number in range(8, 12):
            cache.shard(number)
        assert cache.evictions == 1
        assert len(cache.cache) == 3
        assert cache.admission_skips == 8
        probe.available = 330
        cache.shard(12)
        assert len(cache.cache) == 4
        assert not cache.snapshot()["pressure_active"]
        probe.available = 290
        cache.shard(13)
        assert cache.evictions == 2
        assert cache.pressure_episodes == 2
    finally:
        cache.close()


def test_absolute_cap_and_lru_recency(tmp_path: Path) -> None:
    size = _fixture(tmp_path, shards=3)
    cache = AdaptiveSealedInputs(
        tmp_path,
        policy=_policy(absolute=2 * size),
        memory_probe=Probe(),
    )
    try:
        cache.shard(0)
        cache.shard(1)
        cache.shard(0)
        cache.shard(2)
        assert list(cache.cache) == [0, 2]
        assert cache.cache_bytes <= 2 * size
        assert cache.evictions == 1
    finally:
        cache.close()


def test_duplicate_concurrent_load_is_decoded_and_admitted_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture(tmp_path, shards=1)
    original_decode = AdaptiveSealedInputs._decode
    entered, release = Event(), Event()
    calls = []

    def decode(paths: tuple[Path, Path]) -> dict[str, np.ndarray]:
        calls.append(paths)
        entered.set()
        assert release.wait(timeout=5)
        return original_decode(paths)

    monkeypatch.setattr(AdaptiveSealedInputs, "_decode", staticmethod(decode))
    probe = Probe()
    cache = AdaptiveSealedInputs(tmp_path, policy=_policy(), memory_probe=probe)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(cache.shard, 0) for _ in range(4)]
            assert entered.wait(timeout=5)
            release.set()
            values = [future.result(timeout=5) for future in futures]
        assert all(value is values[0] for value in values)
        assert len(calls) == 1
        assert probe.calls == 1
        assert cache.coalesced_loads == 3
        assert cache.reads == 1
    finally:
        cache.close()


def test_concurrent_admissions_serialize_memory_observations(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=4)

    class ContentionProbe:
        def __init__(self) -> None:
            self.guard = Lock()
            self.active = 0
            self.maximum_active = 0

        def __call__(self) -> MemorySnapshot:
            with self.guard:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            sleep(0.02)
            with self.guard:
                self.active -= 1
            return MemorySnapshot(390, 1_000)

    probe = ContentionProbe()
    cache = AdaptiveSealedInputs(tmp_path, policy=_policy(), memory_probe=probe)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(cache.shard, range(4)))
        assert probe.maximum_active == 1
        assert cache.decisions == 4
    finally:
        cache.close()


def test_load_failure_clears_inflight_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture(tmp_path, shards=1)
    calls = 0

    def fail(paths: tuple[Path, Path]) -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        raise OSError("decode failed")

    monkeypatch.setattr(AdaptiveSealedInputs, "_decode", staticmethod(fail))
    probe = Probe()
    cache = AdaptiveSealedInputs(tmp_path, policy=_policy(), memory_probe=probe)
    try:
        for _ in range(2):
            with pytest.raises(OSError, match="decode failed"):
                cache.shard(0)
        assert calls == cache.load_failures == 2
        assert not cache.inflight
        assert probe.calls == 0
    finally:
        cache.close()


def test_cached_hit_still_checks_sealed_file_identity(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=1)
    cache = AdaptiveSealedInputs(tmp_path, policy=_policy(), memory_probe=Probe())
    try:
        cache.shard(0)
        path = tmp_path / "input/shard_00000.npz"
        with path.open("ab") as stream:
            stream.write(b"x")
        with pytest.raises(ValueError, match="Sealed TRAIN40 shard changed"):
            cache.shard(0)
    finally:
        cache.close()


def test_exported_policy_keeps_original_hard_guard_headroom() -> None:
    assert POLICY == {
        "absolute_cache_bytes": 8 * 1024**3,
        "host_reserve_bytes": 3 * 1024**3,
        "host_trim_target_bytes": int(3.25 * 1024**3),
        "tree_rss_soft_cap_bytes": 15_000_000_000,
    }
