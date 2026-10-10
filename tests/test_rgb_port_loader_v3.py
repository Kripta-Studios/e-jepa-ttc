"""Loader concurrency must not change data, order, validation or RNG."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.rgb_port_loader_v3.loader import DepthTwoEventPrefetch, ParallelGroupRowCache
from operational.rgb_port_revision.cache import GroupRowCache


class Source:
    population_size = 9
    frame_counts = [3] * 9
    identity = {"unchanged": True}

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def batch(self, ids, modality):
        assert modality == "event"
        self.calls.append(list(ids))
        if self.fail in ids:
            raise OSError("simulated storage failure")
        return torch.tensor(ids)


def test_prefetch_preserves_cursor_rng_epoch_and_buffer_ownership():
    source = Source()
    cursor = {"order": torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4]), "position": 0}
    rng = torch.get_rng_state().clone()
    loader = DepthTwoEventPrefetch(source, batch_size=4)
    try:
        loader.bind(cursor)
        assert cursor["position"] == 0
        first = loader.batch([8, 0, 7, 1], "event")
        cursor["position"] = 4
        loader.batch([6, 2, 5, 3], "event")
        cursor["position"] = 8
        loader.batch([4], "event")
        cursor.update(order=torch.arange(9), position=0)
        loader.batch([0, 1, 2, 3], "event")
        assert first.tolist() == [8, 0, 7, 1]
        assert loader.outstanding_max == 2
        assert loader.identity is source.identity
        assert cursor["position"] == 0
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        loader.close()


def test_prefetch_rejects_reordering_and_surfaces_future_failure_on_consumption():
    loader = DepthTwoEventPrefetch(Source(fail=4), batch_size=4)
    try:
        loader.bind({"order": torch.arange(9), "position": 0})
        with pytest.raises(RuntimeError, match="canonical order"):
            loader.batch([3, 2, 1, 0], "event")
        assert loader.batch([0, 1, 2, 3], "event").tolist() == [0, 1, 2, 3]
        with pytest.raises(OSError, match="storage failure"):
            loader.batch([4, 5, 6, 7], "event")
    finally:
        loader.close()


def test_prefetch_warmup_is_synchronous_and_close_is_final():
    source = Source()
    loader = DepthTwoEventPrefetch(source)
    assert loader.batch([5, 1], "event").tolist() == [5, 1]
    assert loader.submitted == 0
    loader.close()
    with pytest.raises(ValueError):
        loader.batch([0], "event")
    with pytest.raises(RuntimeError):
        loader.bind({"order": torch.arange(9), "position": 0})


def test_parallel_decode_keeps_teacher_validation_and_exact_counters(tmp_path):
    (tmp_path / "teacher").mkdir()
    for number in range(3):
        fields = dict(ordinals=np.arange(32) + number * 32, tokens=np.arange(32).astype(str))
        np.savez_compressed(tmp_path / f"shard_{number:05d}.npz", **fields, events=np.ones(32))
        np.savez_compressed(
            tmp_path / "teacher" / f"shard_{number:05d}.npz",
            **fields,
            relation_targets=np.zeros(32),
            relation_valid=np.ones(32),
        )
    inputs = SimpleNamespace(directory=tmp_path, output=tmp_path, reads=0, cache={}, cache_bytes=0)
    cache = ParallelGroupRowCache(SimpleNamespace(_inputs=inputs), workers=2)
    try:
        cache._remaining.extend([0, 1, 2])
        for number in range(3):
            actual = cache._decode(number)
            reference = GroupRowCache._decode(
                SimpleNamespace(
                    inputs=SimpleNamespace(directory=tmp_path, output=tmp_path, reads=0), reads=0
                ),
                number,
            )
            for key in reference:
                np.testing.assert_array_equal(actual[key], reference[key])
        assert cache.outstanding_max == 2
        assert cache.reads == inputs.reads == 3
        np.savez_compressed(
            tmp_path / "teacher/shard_00000.npz",
            ordinals=np.arange(32) + 1,
            tokens=np.arange(32).astype(str),
            relation_targets=np.zeros(32),
            relation_valid=np.ones(32),
        )
        cache._remaining.append(0)
        with pytest.raises(ValueError, match="ordinals differ"):
            cache._decode(0)
    finally:
        cache.close()


def test_producer_binds_after_original_restore_and_retains_checkpoint_hooks(tmp_path, monkeypatch):
    from operational.rgb_port_loader_v3 import producer

    run = tmp_path
    fit = run / "fits/E_A5_MATCHED"
    fit.mkdir(parents=True)
    (fit / "CHECKPOINT_POINTER.json").write_text("{}")
    (run / producer.NAME).write_text("{}")
    checkpoint = fit / "checkpoint_last.pt"
    checkpoint.write_bytes(b"unchanged checkpoint")
    state = SimpleNamespace(completed=17, path=checkpoint)
    cursor = {"order": torch.arange(9), "position": 0}
    calls = []

    class EventSource(producer.training.EventProducerSource, Source):
        def __init__(self):
            Source.__init__(self)

        def batch(self, ids, modality):
            calls.append("batch")
            return Source.batch(self, ids, modality)

    class Cache:
        reads = hits = peak_bytes = 0

        def __init__(self, source, *, workers):
            assert workers == 2

        def close(self):
            calls.append("close")

    source = EventSource()
    recipe = SimpleNamespace(modality="event", microbatch_size=4, effective_batch_size=4)

    def original_restore(*args):
        assert calls == []  # no premature lookahead during restore/prewarm
        calls.append("restored")
        return cursor

    def original_fit(loader, recipe, root, **kwargs):
        assert loader.identity is source.identity
        restored = producer.training.ProducerCheckpoint.restore(state)
        assert restored is cursor
        assert loader.batch([0, 1, 2, 3], "event").tolist() == [0, 1, 2, 3]
        producer.training.ProducerCheckpoint.save(state)
        return {"ok": True}

    monkeypatch.setattr(producer, "validate", lambda run: {})
    monkeypatch.setattr(producer, "ParallelGroupRowCache", Cache)
    monkeypatch.setattr(producer.training.ProducerCheckpoint, "restore", original_restore)
    monkeypatch.setattr(
        producer.training.ProducerCheckpoint, "save", lambda *a, **k: calls.append("saved")
    )
    monkeypatch.setattr(producer.training, "fit_producer", original_fit)
    monkeypatch.setattr(
        producer.continuity,
        "main",
        lambda values: producer.training.fit_producer(source, recipe, run),
    )
    assert producer.main(["--run", str(run), "--fit-id", "E_A5_MATCHED"]) == {"ok": True}
    assert calls[0] == "restored" and calls[-1] == "close"
    assert "saved" in calls and cursor["position"] == 0
    assert checkpoint.read_bytes() == b"unchanged checkpoint"
    assert (fit / "loader_v3_checkpoints/checkpoint_000017.json").is_file()


def test_queue_installs_route_after_parent_without_replacing_ownership(tmp_path, monkeypatch):
    from unittest.mock import patch

    from operational.rgb_port_loader_v3 import queue

    monkeypatch.setattr(queue, "validate", lambda run: {})
    monkeypatch.setattr(queue.sys, "argv", ["queue", "resume", "--run", str(tmp_path)])
    monkeypatch.setattr(
        queue.io_queue.frozen_queue, "_load_config", lambda path: {"package": {"members": []}}
    )

    def original_main(argv):
        assert queue.io_queue.PRODUCER_MODULE == "operational.rgb_port_loader_v3.producer"
        config = queue.io_queue.frozen_queue._load_config(tmp_path)
        assert queue.NAME in config["package"]["members"]
        return 0

    def parent_main():
        with patch.object(
            queue.io_queue, "PRODUCER_MODULE", "operational.rgb_port_continuity.producer"
        ):
            return queue.io_queue.main(["resume", "--run", str(tmp_path)])

    monkeypatch.setattr(queue.io_queue, "main", original_main)
    monkeypatch.setattr(queue.continuity, "main", parent_main)
    assert queue.main() == 0


@pytest.mark.parametrize("margin,expected", [(0.99, False), (1, True), (2.95, True), (3, True)])
def test_requested_commit_reserve_boundaries(margin, expected):
    from operational.rgb_port_loader_v3.memory import with_requested_reserve

    values = {
        "windows_commit_headroom_bytes": int(margin * 1024**3),
        "host_available_bytes": 4 * 1024**3,
        "disk_free_after_reservation_bytes": 20_000_000_000,
    }
    admitted, snapshot = with_requested_reserve((margin >= 3, values))
    assert admitted is expected
    assert snapshot["commit_reserve_bytes"] == 1024**3


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_available_bytes", 1024**3),
        ("disk_free_after_reservation_bytes", 9_999_999_999),
        ("project_tree_rss_bytes", 23_000_000_001),
    ],
)
def test_lower_commit_reserve_does_not_relax_other_gates(field, value):
    from operational.rgb_port_loader_v3.memory import with_requested_reserve

    values = {
        "windows_commit_headroom_bytes": 2 * 1024**3,
        "host_available_bytes": 4 * 1024**3,
        "disk_free_after_reservation_bytes": 20_000_000_000,
        field: value,
    }
    assert with_requested_reserve((False, values))[0] is False


def test_reserve_overlay_rejects_unknown_upstream_gate():
    from operational.rgb_port_loader_v3.memory import with_requested_reserve

    values = {
        "windows_commit_headroom_bytes": 5 * 1024**3,
        "host_available_bytes": 4 * 1024**3,
        "disk_free_after_reservation_bytes": 20_000_000_000,
    }
    with pytest.raises(RuntimeError, match="Upstream resource policy"):
        with_requested_reserve((False, values))
