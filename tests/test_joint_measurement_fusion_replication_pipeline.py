"""Admission barriers must run before opening replica V inputs or checkpoint tensors."""

from pathlib import Path

import pytest

from operational.joint_measurement import fusion_replication_pipeline as pipeline


def test_v_cache_requires_complete_six_fit_graph(monkeypatch):
    monkeypatch.setattr(pipeline, "read", lambda _: {"sources": {}})

    def reject():
        raise ValueError("head still training")

    monkeypatch.setattr(pipeline, "freeze_all", reject)
    monkeypatch.setattr(pipeline, "SensorDataset", lambda _: pytest.fail("V opened too early"))
    with pytest.raises(ValueError, match="head still training"):
        pipeline.cache(13, "V")


@pytest.mark.parametrize("role", ["P", "test12", "FCWD"])
def test_replica_cache_rejects_undeclared_roles(monkeypatch, role):
    monkeypatch.setattr(pipeline, "read", lambda _: {"sources": {}})
    monkeypatch.setattr(pipeline, "SensorDataset", lambda _: pytest.fail("undeclared input opened"))
    with pytest.raises(ValueError, match="Only declared"):
        pipeline.cache(13, role)


@pytest.mark.parametrize("pending,durable", [(1, 12000), (0, 11900)])
def test_producer_rejects_partial_checkpoint_before_tensor_load(monkeypatch, pending, durable):
    monkeypatch.setattr(pipeline, "registered", lambda: {})
    records = {
        "CHECKPOINT_RECEIPT.json": {
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "completed_updates": 12000,
        },
        "UPDATE_JOURNAL.json": {
            "completed_updates": 12000,
            "durable_updates": durable,
            "pending_update_upper": pending,
        },
        "CHECKPOINT_POINTER.json": {"completed_updates": 12000},
        "IDENTITY.json": {"seed": 13},
    }
    monkeypatch.setattr(pipeline, "read", lambda path: records[Path(path).name])
    monkeypatch.setattr(
        pipeline.torch, "load", lambda *a, **k: pytest.fail("partial tensor loaded")
    )
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.producer(13)


def test_replica_identity_binds_seed_specific_producer():
    record = {
        "endpoints": {"13": "a" * 64, "23": "b" * 64},
        "event_schema_sha256": "c" * 64,
        "event_extractor_sha256": "d" * 64,
    }
    a, b = pipeline.feature_identity(13, record), pipeline.feature_identity(23, record)
    assert a.schema_sha256 == b.schema_sha256
    assert a.normalizer_producer_sha256() != b.normalizer_producer_sha256()
