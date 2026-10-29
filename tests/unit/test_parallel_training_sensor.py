"""Exact sampled-batch transport and fail-closed admission for parallel decoding."""

import io
import json
import zipfile
from collections import OrderedDict
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from operational.joint_measurement.data import SensorDataset
from operational.joint_measurement.parallel_training_sensor import ParallelTrainingSensors
from operational.range_transfer.common import sha


@pytest.fixture
def sensor_fixture(tmp_path, monkeypatch):
    records = []
    for index in range(4):
        records.append(
            dict(
                events=np.full((2, 12, 128, 128), index / 7, dtype=np.float32),
                delta=np.asarray(0.03 + index / 100, dtype=np.float32),
                target_ttc=np.asarray(index - 2.5, dtype=np.float32),
                visible_heights=np.asarray([index + 1, index + 2], dtype=np.float32),
                motion=np.asarray([index, -index], dtype=np.float32),
                boxes=np.full((2, 4), index, dtype=np.float32),
                square=np.asarray([0, 0, 128, 128], dtype=np.float32),
                relation_targets=np.full((1, 3), index / 3, dtype=np.float32),
                relation_valid=np.asarray([index % 2 == 0], dtype=bool),
                token=np.asarray(f"token-{index}"),
                source_ordinal=np.asarray(index + 700),
            )
        )

    def repack(*, duplicate_outer=False, duplicate_inner=False):
        path = tmp_path / "pack_00000.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as pack:
            for index, record in enumerate(records):
                stream = io.BytesIO()
                np.savez(stream, **record)
                if duplicate_inner and index == 0:
                    with zipfile.ZipFile(stream, "a") as inner:
                        inner.writestr("delta.npy", inner.read("delta.npy"))
                pack.writestr(f"{index:06d}.npz", stream.getvalue())
            if duplicate_outer:
                pack.writestr("000000.npz", b"duplicate")
        path.with_suffix(".json").write_text(
            json.dumps({"start": 0, "stop": 4, "sha256": sha(path)}), encoding="utf-8"
        )

    repack()

    def init(self, role, **kwargs):
        self.role, self.folder, self.population = role, tmp_path, 4
        self.rows = pd.DataFrame(
            {
                "sample_token": [f"token-{i}" for i in range(4)],
                "sequence_id": ["sequence-a", "sequence-b", "sequence-a", "sequence-b"],
                "track_id": [10, 11, 12, 13],
            }
        )
        self.packs, self.verified = OrderedDict(), set()

    monkeypatch.setattr(SensorDataset, "__init__", init)
    monkeypatch.setattr(
        "operational.joint_measurement.parallel_training_sensor.psutil.virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    return records, repack, tmp_path


def assert_batch_exact(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), field.name
        else:
            assert a == b, field.name


def test_all_fields_duplicates_and_cache_are_exact(sensor_fixture):
    original, parallel = SensorDataset("P"), ParallelTrainingSensors(workers=2)
    try:
        ids = [3, 0, 3, 1, 2, 0]
        expected = original.batch(ids)
        assert_batch_exact(expected, parallel.batch(ids))
        assert_batch_exact(expected, parallel.batch(ids))
        assert parallel.hits == 4 and parallel.misses == 4
        assert parallel.duplicate_draws == 4
        assert parallel.retained_bytes <= parallel.limit_bytes
    finally:
        original.close()
        parallel.shutdown()
    assert not parallel.packs and not parallel.rows_cached
    with pytest.raises(RuntimeError, match="closed"):
        parallel.batch([0])


def test_prefetch_preserves_sampler_and_global_rng(sensor_fixture):
    original, parallel = SensorDataset("P"), ParallelTrainingSensors(workers=2)
    sampler = torch.Generator().manual_seed(19)
    before = torch.get_rng_state().clone()
    try:
        parallel.prepare(sampler.get_state(), pinned=False)
        for _ in range(3):
            ids = torch.randint(4, (32,), generator=sampler).tolist()
            assert_batch_exact(original.batch(ids), parallel.batch(ids))
        assert torch.equal(before, torch.get_rng_state())
        with pytest.raises(ValueError, match="authoritative sampler"):
            parallel.batch([0])
    finally:
        original.close()
        parallel.shutdown()


@pytest.mark.parametrize("which", ["pack", "receipt"])
def test_mutation_rejected_on_cache_hit(sensor_fixture, which):
    _, _, folder = sensor_fixture
    parallel = ParallelTrainingSensors(workers=2)
    try:
        parallel.batch([0])
        path = folder / ("pack_00000.zip" if which == "pack" else "pack_00000.json")
        with path.open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(ValueError, match="changed"):
            parallel.batch([0])
    finally:
        parallel.shutdown()


@pytest.mark.parametrize("corruption", ["token", "clock", "inner", "outer", "hash"])
def test_reject_corrupt_admission(sensor_fixture, corruption):
    records, repack, folder = sensor_fixture
    if corruption == "token":
        records[0]["token"] = np.asarray("incorrect")
    if corruption == "clock":
        records[0]["delta"] = np.asarray(0, dtype=np.float32)
    if corruption in {"inner", "outer"}:
        with pytest.warns(UserWarning, match="Duplicate"):
            repack(duplicate_inner=corruption == "inner", duplicate_outer=corruption == "outer")
    else:
        repack()
    if corruption == "hash":
        receipt = folder / "pack_00000.json"
        receipt.write_text(json.dumps({"start": 0, "stop": 4, "sha256": "bad"}))
    parallel = ParallelTrainingSensors(workers=2)
    try:
        with pytest.raises(ValueError):
            parallel.batch([0])
    finally:
        parallel.shutdown()


def test_zero_cache_and_concurrent_load_guard(sensor_fixture):
    original, parallel = SensorDataset("P"), ParallelTrainingSensors(workers=1, limit_bytes=0)
    try:
        for ids in ([0, 1, 0], [2, 3]):
            assert_batch_exact(original.batch(ids), parallel.batch(ids))
        assert parallel.retained_bytes == 0 and not parallel.rows_cached
        parallel.load_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="concurrently"):
                parallel.load([0])
        finally:
            parallel.load_lock.release()
        for ids in ([], [-1], [4]):
            with pytest.raises(ValueError, match="index"):
                parallel.batch(ids)
    finally:
        original.close()
        parallel.shutdown()
