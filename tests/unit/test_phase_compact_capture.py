"""Compact hooks preserve forwards, exceptions and physical batch identities."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_timewarp.capture import bind_calls, capture_calls
from operational.phase_timewarp.compact_capture import bind_compact_road, capture_compact
from operational.rgb_port import transfer as engine
from scripts.population_phase_canonical_pair import completed_query


class Pair(nn.Module):
    def forward(self, batch):
        return batch.features[:, 0]


def test_compact_hooks_equal_full_capture_without_sensor_copies(monkeypatch):
    models = {"PAIR_E_MATCHED": Pair(), "PAIR_R": Pair()}
    pixels, delta = torch.ones(2, 3, 12, 4, 4), torch.full((2, 2), 0.1)
    invoked = []

    def expert(models, prefix, sensor, dt):
        invoked.append((id(sensor), id(dt)))
        features = torch.ones(2, 133)
        phase = models["PAIR_E_MATCHED"](SimpleNamespace(features=features))
        return phase[:, None].repeat(1, 17), phase[:, None].repeat(1, 3)

    monkeypatch.setattr(engine, "_expert_block", expert)
    with capture_calls(models, "cpu") as (full_calls, full):
        expected = engine._expert_block(models, "E", pixels, delta)
    with capture_compact(models) as (calls, arrays):
        actual = engine._expert_block(models, "E", pixels, delta)
    assert invoked == [(id(pixels), id(delta))] * 2
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    assert set(arrays) == set(full) - {"call0.pixels"}
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, full[key])
    items = [
        {
            "anchor_us": i * 100000 + 300000,
            "timestamps_us": [i * 100000 + t for t in (100000, 200000, 300000)],
        }
        for i in range(2)
    ]
    candidates = {"event": items, "rgb": []}
    bind_calls(full_calls, full, candidates, candidates)
    bind_compact_road(calls, arrays, candidates, candidates)
    assert calls[0]["observations"] == full_calls[0]["observations"]
    assert not models["PAIR_E_MATCHED"]._forward_hooks
    assert engine._expert_block is expert


def test_failed_forward_restores_hooks_and_retains_failure(monkeypatch):
    models = {"PAIR_E_MATCHED": Pair(), "PAIR_R": Pair()}

    def fail(*args):
        raise ValueError("upstream failed")

    monkeypatch.setattr(engine, "_expert_block", fail)
    with capture_compact(models) as (calls, arrays), pytest.raises(ValueError, match="upstream"):
        engine._expert_block(models, "E", torch.ones(1, 2, 1), torch.ones(1, 1))
    assert calls[0]["status"] == "rejected"
    assert list(arrays) == ["call0.delta"]
    assert engine._expert_block is fail
    assert not models["PAIR_R"]._forward_hooks


def test_resume_rejects_other_protocol_or_changed_predictions(tmp_path):
    record = {"token": "road:1", "group": "road", "baseline_exact": True}
    atomic_receipt(tmp_path / "RESULT.json", record)
    atomic_receipt(
        tmp_path / "COMPLETE.json",
        {
            **record,
            "freeze_sha256": "protocol-a",
            "artifacts": {"RESULT.json": file_sha(tmp_path / "RESULT.json")},
        },
    )
    assert completed_query(tmp_path, "road:1", "road", "protocol-a") == record
    with pytest.raises(ValueError, match="binding"):
        completed_query(tmp_path, "road:1", "road", "protocol-b")
    (tmp_path / "RESULT.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        completed_query(tmp_path, "road:1", "road", "protocol-a")
