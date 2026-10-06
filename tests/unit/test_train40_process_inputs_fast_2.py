"""CPU-only parity, failure and cleanup tests for spawned TRAIN40 inputs."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from operational.train40_system.engine_overlap import SealedInputs
from operational.train40_system.process_inputs_fast_2 import ProcessInputs


def _fixture(output: Path, *, shards: int = 16, size: int | None = None) -> None:
    cache = output / "input"
    teacher = output / "teacher"
    cache.mkdir(parents=True)
    teacher.mkdir()
    rows: list[dict[str, str]] = []
    input_entries = []
    teacher_entries = []
    for number in range(shards):
        start = number * 32
        count = 32 if size is None else min(32, max(0, size - start))
        if count == 0:
            break
        ordinals = np.arange(start, start + count, dtype=np.int64)
        tokens = np.asarray([f"token-{value}" for value in ordinals])
        rng = np.random.default_rng(200 + number)
        values = {
            "events": rng.normal(size=(count, 3, 12, 8, 8)).astype(np.float32),
            "delta": np.full(count, 0.1, dtype=np.float32),
            "motion": rng.normal(size=(count, 18)).astype(np.float32),
            "visible_heights": np.full((count, 2), [8.0, 9.0], dtype=np.float32),
            "target_ttc": np.full(count, -2.0, dtype=np.float32),
            "boxes": np.tile(
                np.asarray([[0, 0, 4, 4], [1, 1, 5, 5], [2, 2, 6, 6]], np.float32),
                (count, 1, 1),
            ),
            "square": np.tile(np.asarray([0, 0, 8, 8], np.float32), (count, 1)),
            "ordinals": ordinals,
            "tokens": tokens,
        }
        relations = rng.normal(size=(count, 2, 6, 2, 2)).astype(np.float16)
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
        for path, entries in ((input_path, input_entries), (teacher_path, teacher_entries)):
            stat = path.stat()
            entries.append(
                {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
        rows.extend(
            {
                "sequence_id": "sequence",
                "sample_token": str(token),
                "track_id": f"track-{ordinal}",
            }
            for ordinal, token in zip(ordinals, tokens, strict=True)
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


def _assert_batches_equal(left: Any, right: Any) -> None:
    for name in left.__dataclass_fields__:
        first, second = getattr(left, name), getattr(right, name)
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second), name
        else:
            assert first == second, name


def test_spawned_batches_b32_b8_fifo_lookahead_rng_and_cleanup(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=18, size=552)
    cpu_rng = torch.get_rng_state().clone()
    original = SealedInputs(tmp_path)
    loader = ProcessInputs(tmp_path, slots=2, slot_bytes=2 * 1024**2, timeout_seconds=60)
    names = [arena.name for arena in loader._arenas]
    try:
        order = torch.arange(512, dtype=torch.int64)
        # The main training thread may publish lookahead while the prefetch
        # thread owns the IPC lock. This must remain a local, nonblocking hint.
        with loader._lock:
            loader.prepare_order(order, 0, 32)
            assert loader.snapshot()["requested_batch_count"] == 0
        ids32 = list(range(31, -1, -1))
        ids8 = [256, 263, 271, 279, 287, 295, 303, 319]
        _assert_batches_equal(loader.batch(ids32), original.batch(ids32))
        _assert_batches_equal(loader.batch(ids8), original.batch(ids8))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(loader.batch, ids) for ids in (ids32, ids8)]
            actual = [future.result(timeout=60) for future in futures]
        _assert_batches_equal(actual[0], original.batch(ids32))
        _assert_batches_equal(actual[1], original.batch(ids8))
        group_requests_before = loader.snapshot()["group_requests"]
        eight_shards = list(range(0, 256, 32))
        _assert_batches_equal(loader.batch(eight_shards), original.batch(eight_shards))
        assert loader.snapshot()["group_requests"] - group_requests_before == 1
        # A 40-row final logical group may occur in the middle of the shuffled
        # epoch. A B32 then legitimately spans that short group and its successor.
        short_first = torch.cat((torch.arange(512, 552), torch.arange(512)))
        loader.prepare_order(short_first, 24, 32)
        boundary_ids = short_first[24:56].tolist()
        _assert_batches_equal(loader.batch(boundary_ids), original.batch(boundary_ids))
        snapshot = loader.snapshot()
        assert set(snapshot["cache_groups"]) == {0, 2}
        assert snapshot["reads"] == 18
        assert snapshot["hits"] > 0
        assert snapshot["cache_bytes"] <= snapshot["cache_budget_bytes"]
        assert snapshot["decoded_groups"] == 3
        assert snapshot["decode_seconds"] > 0
        assert snapshot["collate_seconds"] > 0
        assert snapshot["arena_write_seconds"] > 0
        assert snapshot["parent_ipc_wait_seconds"] > 0
        assert snapshot["parent_reconstruct_seconds"] > 0
        assert snapshot["requested_batch_count"] == 6
        assert not snapshot["poisoned"]
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        path = tmp_path / "input/shard_00000.npz"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with pytest.raises(RuntimeError, match="Sealed TRAIN40 shard changed"):
            loader.batch(list(range(8)))
        with pytest.raises(RuntimeError, match="poisoned"):
            loader.batch(list(range(8)))
        assert loader.snapshot()["poisoned"]
    finally:
        loader.close()
        original.close()
    closed = loader.snapshot()
    assert closed["closed"] and not closed["worker_alive"]
    assert closed["cache_bytes"] >= 0
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


def test_worker_death_poison_and_unlinks_shared_arenas(tmp_path: Path) -> None:
    _fixture(tmp_path, shards=8)
    loader = ProcessInputs(tmp_path, slots=2, slot_bytes=2 * 1024**2, timeout_seconds=5)
    names = [arena.name for arena in loader._arenas]
    loader._process.terminate()
    loader._process.join(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="exited"):
            loader.batch(list(range(8)))
        with pytest.raises(RuntimeError, match="poisoned"):
            loader.batch(list(range(8)))
    finally:
        loader.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


def test_source_has_no_accelerator_or_optimizer_calls() -> None:
    source = Path(ProcessInputs.__module__.replace(".", "/") + ".py")
    if not source.is_file():
        source = (
            Path(__file__).parents[2]
            / "operational/train40_system/process_inputs_fast_2.py"
        )
    text = source.read_text(encoding="utf-8")
    assert "torch.cuda" not in text
    assert "optimizer" not in text
    assert 'get_context("spawn")' in text


def test_variant_changes_only_worker_thread_count() -> None:
    root = Path(__file__).parents[2] / "operational/train40_system"
    base = (root / "process_inputs_fast_collate.py").read_bytes()
    candidate = (root / "process_inputs_fast_2.py").read_bytes()
    before = b"torch.set_num_threads(1)"
    after = b"torch.set_num_threads(2)"
    assert base.count(before) == 1
    assert candidate == base.replace(before, after)
