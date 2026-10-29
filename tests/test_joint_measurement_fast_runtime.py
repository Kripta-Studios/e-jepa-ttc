"""Lossless cache and speculative-sampler contracts for the runtime overlay."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from operational.joint_measurement import data, train
from operational.joint_measurement.fast_runtime import PersistentSensors, batched_backward
from operational.joint_measurement.qa import exact, fixture, subset
from operational.range_transfer.common import sha, write


@pytest.fixture
def sensors(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        {"sample_token": ["a", "b"], "sequence_id": ["s", "s"], "track_id": ["t", "t"]}
    )
    path = tmp_path / "pack_00000.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for index, token in enumerate(("a", "b")):
            buffer = io.BytesIO()
            np.savez(
                buffer,
                token=np.array(token),
                events=np.full((2, 12, 8, 8), index, dtype=np.float32),
                delta=np.array(0.1, dtype=np.float32),
                motion=np.zeros(18, np.float32),
                visible_heights=np.ones(2, np.float32),
                target_ttc=np.array(1.0, np.float32),
                boxes=np.zeros((2, 4), np.float32),
                square=np.zeros(4, np.float32),
                relation_targets=np.zeros((2, 6, 2, 2), np.float32),
                relation_valid=np.ones((2, 6, 2, 2), bool),
            )
            out.writestr(f"{index:06d}.npz", buffer.getvalue())
    write(path.with_suffix(".json"), {"sha256": sha(path)})
    write(tmp_path / "COMPLETE.json", {"rows": 2})
    monkeypatch.setattr(data, "cache_root", lambda role: tmp_path)
    monkeypatch.setattr(data, "role_rows", lambda role: rows)
    reader = PersistentSensors(limit_bytes=1024**2)
    yield reader
    reader.shutdown()


def test_cache_preserves_duplicates_and_scopes(sensors):
    original = data.SensorDataset("P")
    try:
        left, right = original.batch([1, 0, 1]), sensors.batch([1, 0, 1])
        assert exact(vars(left), vars(right))
        sensors.close()
        assert sensors.verified == {0}
        assert sensors.counters()["retained_bytes"] > 0
        assert exact(vars(right), vars(sensors.batch([1, 0, 1])))
    finally:
        original.close()


def test_mutation_invalidates_even_ram_cache(sensors):
    sensors.batch([0])
    with (sensors.folder / "pack_00000.json").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="changed"):
        sensors.batch([0])


def test_prefetch_preserves_rng_and_resume(sensors):
    sampler = torch.Generator().manual_seed(123)
    before = sampler.get_state().clone()
    sensors.prepare(before, pinned=False)
    assert torch.equal(before, sampler.get_state())
    ids = torch.randint(2, (32,), generator=sampler).tolist()
    result = sensors.batch(ids)
    assert result.sample_tokens == [["a", "b"][i] for i in ids]
    saved = sampler.get_state().clone()
    sensors.close()
    assert torch.equal(saved, sampler.get_state())
    sensors.prepare(saved, pinned=False)
    ids = torch.randint(2, (32,), generator=sampler).tolist()
    assert sensors.batch(ids).sample_tokens == [["a", "b"][i] for i in ids]


def test_cache_obeys_byte_bound(sensors):
    sensors.limit_bytes = 1
    sensors.batch([0, 1])
    assert sensors.retained_bytes == 0


def test_grouped_metrics_preserve_native_loss_gradient_and_rng():
    train.setup(7)
    model = train.initialize("NATIVE-P2-FT-S7", random_parent=True)
    batch = fixture()
    batches = [subset(batch, [0, 1]), subset(batch, [2, 3])]
    state = torch.get_rng_state()
    expected = train.backward(model, batches, None, 8)
    gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    model.zero_grad(set_to_none=True)
    actual = batched_backward(model, batches, None, 8)
    assert expected == actual
    assert exact(gradients, {k: p.grad for k, p in model.named_parameters() if p.grad is not None})
    assert torch.equal(after, torch.get_rng_state())
