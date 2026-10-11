"""Checkpoint and wrapper contracts for the additive RGB overlap runtime."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_event_route_delegates_without_rgb_admission(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_overlap import producer

    monkeypatch.setattr(producer, "validate", lambda run: pytest.fail("RGB admission invoked"))
    monkeypatch.setattr(producer.parent, "main", lambda values: 17)
    assert producer.main(["--run", str(tmp_path), "--fit-id", "E_A5_MATCHED"]) == 17


@pytest.mark.parametrize("mode", ["resume", "fresh_positional", "fresh_keyword"])
@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_binding_preserves_cursor_and_teardown(monkeypatch, tmp_path, mode, fail):
    _assert_checkpoint_runtime(monkeypatch, tmp_path, mode, fail, compute_enabled=True)


def test_disabled_geometry_never_installs_compute_patch(monkeypatch, tmp_path):
    _assert_checkpoint_runtime(monkeypatch, tmp_path, "resume", False, compute_enabled=False)


def _assert_checkpoint_runtime(monkeypatch, tmp_path, mode, fail, *, compute_enabled):
    from operational.rgb_port_rgb_overlap import producer

    fit_id = "R_A5" if mode == "resume" else "R_C2F"
    fit = tmp_path / "fits" / fit_id
    fit.mkdir(parents=True)
    checkpoint = fit / "checkpoint.pt"
    checkpoint.write_bytes(b"unchanged native checkpoint")
    (tmp_path / producer.NAME).write_text("{}")
    state = SimpleNamespace(path=checkpoint, completed=1500 if mode == "resume" else 0)
    cursor = {"position": 64, "epoch": 1, "order": [3, 1, 2, 0]}
    source = SimpleNamespace(identity={"preserved": True})
    recipe = SimpleNamespace(modality="rgb", effective_batch_size=32)
    events = []
    bindings = []

    @contextmanager
    def compute():
        assert compute_enabled, "Disabled geometry must not enter the compute context"
        events.append("compute_open")
        try:
            yield SimpleNamespace(snapshot=lambda: {"active": True})
        finally:
            events.append("compute_closed")

    @contextmanager
    def decoder(value, **kwargs):
        assert value is source
        assert kwargs == {"workers": 4, "max_pending_bytes": 128 * 1024**2}
        events.append("decoder_open")
        try:
            yield SimpleNamespace(snapshot=lambda: {"reads": 2})
        finally:
            events.append("decoder_closed")

    @contextmanager
    def prefetch(effective_type, **kwargs):
        assert effective_type is producer.EffectiveSource
        assert kwargs == {"expected_source": source, "batch_size": 32, "depth": 2}
        events.append("prefetch_open")

        def bind(value):
            assert value is cursor
            bindings.append(value)
            events.append("bind")

        try:
            yield SimpleNamespace(bind=bind, snapshot=lambda: {"bound": bool(bindings)})
        finally:
            events.append("prefetch_closed")

    def original_restore(value, *args, **kwargs):
        assert value is state
        events.append("native_restored")
        return cursor

    def original_save(value, *args, **kwargs):
        assert value is state
        events.append("native_saved")

    native_args = (object(), object(), object(), object(), cursor)

    def original_fit(value, actual_recipe, run, **kwargs):
        assert value is source and actual_recipe is recipe and run == tmp_path
        assert kwargs == {"device": "cpu"}
        assert bindings == []
        if mode == "resume":
            assert producer.training.ProducerCheckpoint.restore(state, *native_args[:4]) is cursor
        elif mode == "fresh_positional":
            producer.training.ProducerCheckpoint.save(state, *native_args, status="READY")
        else:
            producer.training.ProducerCheckpoint.save(
                state, *native_args[:4], cursor=cursor, status="READY"
            )
        assert bindings == [cursor]
        producer.training.ProducerCheckpoint.save(state, *native_args, status="PAUSED_RESOURCE")
        assert bindings == [cursor], "Later checkpoint saves must not rebind lookahead"
        if fail:
            raise OSError("simulated input outage")
        return 3

    def parent_main(values):
        # Historical parent source patches are already active when the inner
        # fit executes; overlap must keep that raw source, not wrap its identity.
        events.append("parent_open")
        try:
            return producer.training.fit_producer(source, recipe, tmp_path, device="cpu")
        finally:
            events.append("parent_closed")

    monkeypatch.setattr(producer, "validate", lambda run: {})
    monkeypatch.setattr(producer, "COMPUTE_ENABLED", compute_enabled)
    monkeypatch.setattr(producer, "installed_compute", compute)
    monkeypatch.setattr(producer, "installed_decoder", decoder)
    monkeypatch.setattr(producer, "capture_effective_prefetch", prefetch)
    monkeypatch.setattr(producer.training, "fit_producer", original_fit)
    monkeypatch.setattr(producer.training.ProducerCheckpoint, "restore", original_restore)
    monkeypatch.setattr(producer.training.ProducerCheckpoint, "save", original_save)
    monkeypatch.setattr(producer.parent, "main", parent_main)
    argv = ["--run", str(tmp_path), "--fit-id", fit_id]
    if fail:
        with pytest.raises(OSError, match="input outage"):
            producer.main(argv)
    else:
        assert producer.main(argv) == 3
    closing = ["prefetch_closed", "decoder_closed"]
    if compute_enabled:
        closing.append("compute_closed")
    closing.append("parent_closed")
    assert events[-len(closing) :] == closing
    boundary = "native_restored" if mode == "resume" else "native_saved"
    assert events.index(boundary) < events.index("bind")
    assert producer.training.fit_producer is original_fit
    assert producer.training.ProducerCheckpoint.restore is original_restore
    assert producer.training.ProducerCheckpoint.save is original_save
    assert checkpoint.read_bytes() == b"unchanged native checkpoint"
    assert cursor == {"position": 64, "epoch": 1, "order": [3, 1, 2, 0]}
    runtime = json.loads((fit / "RGB_OVERLAP_RUNTIME.json").read_text())
    assert runtime["origin"]["completed_updates"] == state.completed
    assert runtime["objective_changed"] is False
    assert runtime["geometry"]["enabled"] is compute_enabled
    assert runtime["prefetch"]["bound"] is True
    origin = json.loads(
        (fit / "rgb_overlap_checkpoints" / f"origin_{state.completed:06d}.json").read_text()
    )
    assert origin["status"] == ("RESTORED" if mode == "resume" else "INITIALIZED")


def test_queue_final_route_and_complete_test_packaging(monkeypatch, tmp_path):
    from operational.rgb_port_rgb_overlap import queue

    monkeypatch.setattr(queue, "validate", lambda run: {})
    monkeypatch.setattr(queue.sys, "argv", ["queue", "resume", "--run", str(tmp_path)])
    monkeypatch.setattr(
        queue.io_queue.frozen_queue, "_load_config", lambda path: {"package": {"members": []}}
    )

    def inner_main(argv):
        assert queue.io_queue.PRODUCER_MODULE == "operational.rgb_port_rgb_overlap.producer"
        members = queue.io_queue.frozen_queue._load_config(tmp_path)["package"]["members"]
        assert queue.NAME in members
        assert "repo-tree:operational/rgb_port_rgb_overlap" in members
        for path in (queue.ROOT / "tests").glob("test_rgb_port_rgb_overlap*.py"):
            assert "repo:" + path.relative_to(queue.ROOT).as_posix() in members
        return 0

    def parent_main():
        with patch.object(
            queue.io_queue, "PRODUCER_MODULE", "operational.rgb_port_rgb_loader.producer"
        ):
            return queue.io_queue.main(["resume", "--run", str(tmp_path)])

    monkeypatch.setattr(queue.io_queue, "main", inner_main)
    monkeypatch.setattr(queue.parent, "main", parent_main)
    assert queue.main() == 0


@pytest.mark.parametrize(
    "missing", ["exact_batch_parity", "exact_gpu_gradient_parity", "cursor_rng_unchanged", "status"]
)
def test_admission_requires_every_parity_gate(monkeypatch, tmp_path, missing):
    from operational.rgb_port_rgb_overlap import contracts

    quality = {
        "status": "PASSED",
        "exact_batch_parity": True,
        "exact_gpu_gradient_parity": True,
        "cursor_rng_unchanged": True,
    }
    del quality[missing]
    qa = tmp_path / "QA.json"
    qa.write_text(json.dumps(quality))
    monkeypatch.setattr(contracts, "validate_parent", lambda run: {})
    with pytest.raises(ValueError, match="admission requires|QA did not pass"):
        contracts.payload(tmp_path, qa)
