"""Integration contracts for RGB input hooks and inherited recovery routing."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_non_rgb_producer_delegates_without_installing_rgb_runtime(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_loader import producer

    monkeypatch.setattr(producer, "validate", lambda run: pytest.fail("RGB validation invoked"))
    monkeypatch.setattr(producer.parent, "main", lambda values: 17)
    assert producer.main(["--run", str(tmp_path), "--fit-id", "E_A5_MATCHED"]) == 17


def test_rgb_hooks_preserve_source_cursor_checkpoint_and_read_retries(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_loader import producer

    fit = tmp_path / "fits/R_A5"
    fit.mkdir(parents=True)
    checkpoint = fit / "checkpoint.pt"
    checkpoint.write_bytes(b"original state")
    (tmp_path / producer.NAME).write_text("{}")
    state = SimpleNamespace(path=checkpoint, completed=17)
    cursor = {"position": 32}
    calls = []

    class Source:
        identity = {"unchanged": True}

        def batch(self, ids, modality):
            return (ids, modality)

    source = Source()
    original_batch = source.batch
    recipe = SimpleNamespace(modality="rgb")

    @contextmanager
    def cache(value, **kwargs):
        assert value is source
        assert kwargs == {"max_shards": 32, "max_bytes": 64 * 1024**2}
        calls.append("cache_open")
        try:
            yield SimpleNamespace(snapshot=lambda: {"reads": 2})
        finally:
            calls.append("cache_closed")

    @contextmanager
    def collate(value):
        assert value is source
        calls.append("collate_open")
        try:
            yield
        finally:
            calls.append("collate_closed")

    def retry(read, record, *, scope):
        assert scope == "rgb_optimized_producer_batch"
        calls.append("retry")
        return read()

    def original_fit(value, actual_recipe, run, **kwargs):
        assert value is source and actual_recipe is recipe and run == tmp_path
        restored = producer.training.ProducerCheckpoint.restore(state)
        assert restored is cursor
        assert source.batch([2, 9], "rgb") == ([2, 9], "rgb")
        producer.training.ProducerCheckpoint.save(state)
        return 3

    monkeypatch.setattr(producer, "RGBProducerSource", Source)
    monkeypatch.setattr(producer, "validate", lambda run: {})
    monkeypatch.setattr(producer, "installed_dino_cache", cache)
    monkeypatch.setattr(producer, "installed_collate", collate)
    monkeypatch.setattr(producer, "retry_input_read", retry)
    monkeypatch.setattr(producer.training, "fit_producer", original_fit)
    monkeypatch.setattr(producer.training.ProducerCheckpoint, "restore", lambda *a, **kw: cursor)
    monkeypatch.setattr(
        producer.training.ProducerCheckpoint, "save", lambda *a, **kw: calls.append("saved")
    )
    monkeypatch.setattr(
        producer.parent,
        "main",
        lambda values: producer.training.fit_producer(source, recipe, tmp_path),
    )
    assert producer.main(["--run", str(tmp_path), "--fit-id", "R_A5"]) == 3
    assert cursor == {"position": 32}
    assert source.batch == original_batch and "batch" not in source.__dict__
    assert checkpoint.read_bytes() == b"original state"
    assert calls == [
        "cache_open",
        "collate_open",
        "retry",
        "saved",
        "collate_closed",
        "cache_closed",
    ]
    import json

    runtime = json.loads((fit / "RGB_LOADER_RUNTIME.json").read_text())
    assert runtime["source_calls"] == 1 and runtime["origin"]["completed_updates"] == 17
    assert runtime["commit_reserve_bytes"] == 1024**3
    assert (fit / "rgb_loader_checkpoints/origin_000017.json").is_file()


def test_queue_final_route_wins_over_parent_and_extends_package(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_loader import queue

    monkeypatch.setattr(queue, "validate", lambda run: {})
    monkeypatch.setattr(queue.sys, "argv", ["queue", "resume", "--run", str(tmp_path)])
    monkeypatch.setattr(
        queue.io_queue.frozen_queue, "_load_config", lambda path: {"package": {"members": []}}
    )

    def inner_main(argv):
        assert queue.io_queue.PRODUCER_MODULE == "operational.rgb_port_rgb_loader.producer"
        members = queue.io_queue.frozen_queue._load_config(tmp_path)["package"]["members"]
        assert queue.NAME in members
        return 0

    def parent_main():
        with patch.object(
            queue.io_queue, "PRODUCER_MODULE", "operational.rgb_port_loader_v3.producer"
        ):
            return queue.io_queue.main(["resume", "--run", str(tmp_path)])

    monkeypatch.setattr(queue.io_queue, "main", inner_main)
    monkeypatch.setattr(queue.parent, "main", parent_main)
    assert queue.main() == 0


def test_rgb_admission_rejects_qa_without_exact_parity(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_loader import contracts

    qa = tmp_path / "qa.json"
    qa.write_text('{"status": "PASSED", "exact_batch_parity": false}')
    monkeypatch.setattr(contracts, "validate_parent", lambda run: {})
    with pytest.raises(ValueError, match="exact real batch parity"):
        contracts.payload(tmp_path, qa)
