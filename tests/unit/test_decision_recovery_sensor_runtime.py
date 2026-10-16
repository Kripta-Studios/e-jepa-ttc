"""Exactness and resource contracts for the V19 sensor runtime."""

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

from operational.decision_recovery.sensor_runtime import (
    EVALUATION,
    QA,
    TRAINING,
    RoleParallelSensors,
    make_evaluation_sensors,
    make_training_sensors,
    runtime_contract,
)
from operational.joint_measurement.data import SensorDataset
from operational.range_transfer.common import sha


@pytest.fixture
def sensor_fixture(tmp_path, monkeypatch):
    records = []
    for index in range(4):
        records.append(
            {
                "events": np.full((2, 12, 128, 128), index / 9, dtype=np.float32),
                "delta": np.asarray(0.02 + index / 100, dtype=np.float32),
                "target_ttc": np.asarray(index - 2.5, dtype=np.float32),
                "visible_heights": np.asarray([index + 1, index + 2], dtype=np.float32),
                "motion": np.asarray([index, -index], dtype=np.float32),
                "boxes": np.full((2, 4), index, dtype=np.float32),
                "square": np.asarray([0, 0, 128, 128], dtype=np.float32),
                "relation_targets": np.full((1, 3), index / 3, dtype=np.float32),
                "relation_valid": np.asarray([index % 2 == 0], dtype=bool),
                "token": np.asarray(f"token-{index}"),
                "source_ordinal": np.asarray(index + 500),
            }
        )
    path = tmp_path / "pack_00000.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as pack:
        for index, record in enumerate(records):
            stream = io.BytesIO()
            np.savez(stream, **record)
            pack.writestr(f"{index:06d}.npz", stream.getvalue())
    path.with_suffix(".json").write_text(
        json.dumps({"start": 0, "stop": 4, "sha256": sha(path)}), encoding="utf-8"
    )

    def init(self, role, **kwargs):
        self.role, self.folder, self.population = role, tmp_path, 4
        self.rows = pd.DataFrame(
            {
                "sample_token": [f"token-{i}" for i in range(4)],
                "sequence_id": ["s0", "s1", "s0", "s1"],
                "track_id": [20, 21, 22, 23],
            }
        )
        self.packs, self.verified = OrderedDict(), set()

    monkeypatch.setattr(SensorDataset, "__init__", init)
    monkeypatch.setattr(
        "operational.joint_measurement.parallel_training_sensor.psutil.virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    return records


def assert_batch_exact(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), field.name
        else:
            assert a == b, field.name


@pytest.mark.parametrize("role", ["P", "H"])
def test_role_reader_is_exact_and_preserves_duplicates(sensor_fixture, role):
    original = SensorDataset(role)
    parallel = RoleParallelSensors(role, config=EVALUATION)
    try:
        ids = [3, 0, 3, 1]
        expected = original.batch(ids)
        before = torch.get_rng_state().clone()
        assert_batch_exact(expected, parallel.batch(ids))
        assert torch.equal(before, torch.get_rng_state())
        assert parallel.duplicate_draws == 1
        assert parallel.retained_bytes == 0
        assert parallel.counters()["role"] == role
    finally:
        original.close()
        parallel.shutdown()


def test_training_factory_prefetch_preserves_sampler(sensor_fixture):
    original = SensorDataset("P")
    parallel = make_training_sensors(qa=True)
    sampler = torch.Generator().manual_seed(17)
    before = torch.get_rng_state().clone()
    try:
        parallel.prepare(sampler.get_state(), pinned=False)
        ids = torch.randint(4, (32,), generator=sampler).tolist()
        assert_batch_exact(original.batch(ids), parallel.batch(ids))
        assert torch.equal(before, torch.get_rng_state())
        assert parallel.counters()["runtime_config"] == QA.as_dict()
    finally:
        original.close()
        parallel.shutdown()


def test_factories_and_contract_are_fixed(sensor_fixture):
    training = make_training_sensors()
    evaluation = make_evaluation_sensors("H")
    try:
        assert training.runtime_config == TRAINING
        assert evaluation.runtime_config == EVALUATION
        assert TRAINING.workers == QA.workers == EVALUATION.workers == 8
        assert TRAINING.cache_bytes == 4 * 1024**3
        assert QA.cache_bytes == EVALUATION.cache_bytes == 0
        contract = runtime_contract()
        assert contract["training"] == TRAINING.as_dict()
        assert "no transcode" in contract["compression"]
    finally:
        training.shutdown()
        evaluation.shutdown()


@pytest.mark.parametrize("role", ["V", "", "p"])
def test_unregistered_roles_fail_before_read(sensor_fixture, role):
    with pytest.raises(ValueError, match="only P or H"):
        RoleParallelSensors(role, config=EVALUATION)
