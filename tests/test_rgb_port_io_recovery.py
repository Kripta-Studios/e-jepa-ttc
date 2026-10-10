"""Real decode parity, zero-update checkpoint pauses and queue recovery contracts."""

from __future__ import annotations

import errno
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_io_recovery import common, producer, queue


def test_retry_reopens_same_read_and_returns_original_value():
    expected = object()
    events, sleeps = [], []
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError(errno.EINVAL, "disconnected input")
        return expected

    assert (
        common.retry_input_read(read, events.append, scope="shard", sleep=sleeps.append) is expected
    )
    assert calls == 3 and sleeps == [2.0, 2.0]
    assert [v["attempt"] for v in events] == [1, 2]


def test_exhausted_read_enters_existing_graceful_pause_classification():
    from operational.rgb_port.train_producers import _transient_input_error

    events = []

    def read():
        raise OSError(errno.EINVAL, "input")

    with pytest.raises(OSError) as caught:
        common.retry_input_read(read, events.append, scope="shard", sleep=lambda _: None)
    assert caught.value.errno == errno.ENOTCONN
    assert _transient_input_error(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert len(events) == 3 and events[-1]["action"] == "full_state_pause"


@pytest.mark.parametrize(
    "error", [ValueError("corrupt array"), OSError(errno.EACCES, "permissions")]
)
def test_unrelated_or_corrupt_inputs_propagate_without_retry(error):
    events = []

    def read():
        raise error

    with pytest.raises(type(error)) as caught:
        common.retry_input_read(read, events.append, scope="shard", sleep=lambda _: None)
    assert caught.value is error and events == []


def test_actual_npz_decode_retry_is_bit_exact_and_counts_one_success(tmp_path, monkeypatch):
    from operational.rgb_port_revision.cache import GroupRowCache

    inputs = tmp_path / "inputs"
    teacher = tmp_path / "teacher"
    inputs.mkdir()
    teacher.mkdir()
    expected = {
        "ordinals": np.arange(4),
        "tokens": np.array(["a", "b", "c", "d"]),
        "events": np.arange(48, dtype=np.float32).reshape(4, 3, 4),
    }
    targets = {
        "ordinals": expected["ordinals"],
        "tokens": expected["tokens"],
        "relation_targets": np.arange(8, dtype=np.float32).reshape(4, 2),
        "relation_valid": np.ones((4, 2), dtype=bool),
    }
    np.savez_compressed(inputs / "shard_00000.npz", **expected)
    np.savez_compressed(teacher / "shard_00000.npz", **targets)
    source = SimpleNamespace(
        _inputs=SimpleNamespace(
            directory=inputs, output=tmp_path, cache=OrderedDict(), cache_bytes=0, reads=0, hits=0
        )
    )
    cache = GroupRowCache(source)
    original_load = np.load
    calls = 0

    def interrupted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EINVAL, "interrupted external drive")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", interrupted_load)
    monkeypatch.setattr(
        producer,
        "retry_input_read",
        lambda *a, **kw: common.retry_input_read(*a, **kw, sleep=lambda _: None),
    )
    with producer.installed(tmp_path, "E_A5_MATCHED"):
        decoded = cache._decode(0)
    for key, value in {**expected, **targets}.items():
        np.testing.assert_array_equal(decoded[key], value)
    assert cache.reads == source._inputs.reads == 1
    assert len(list((tmp_path / "io_recovery/E_A5_MATCHED").glob("*.json"))) == 1


def test_input_failure_saves_full_checkpoint_before_any_optimizer_update(tmp_path, monkeypatch):
    from operational.rgb_port import train_producers as training
    from operational.rgb_port.recipe import resolved_recipe
    from operational.rgb_port_revision import migration

    recipe = resolved_recipe(
        Path("configs/rgb_port/producers.json"),
        fit_id="E_A5_MATCHED",
        producer_population=1,
        role_manifest_sha256="a" * 64,
        microbatch_size=1,
    )
    atomic_write_json(tmp_path / "SOURCE_FREEZE.json", {"base_commit": "zero_update_io_test"})
    monkeypatch.setattr(migration, "bind_runtime", lambda *_a, **_k: "b" * 64)
    model = torch.nn.Linear(2, 1)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(training, "build_producer", lambda _: model)
    monkeypatch.setattr(training, "_resource_guard", lambda *_a, **_k: (True, {}))
    monkeypatch.setattr(training, "_fast_resource_guard", lambda *_a, **_k: (True, {}))

    def forbidden_update(*_a, **_k):
        raise AssertionError("This QA must consume zero optimizer updates")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)

    def unavailable(_indices, _modality):
        def read():
            raise OSError(errno.EINVAL, "external input unavailable")

        return common.retry_input_read(read, lambda _: None, scope="fixture", sleep=lambda _: None)

    source = SimpleNamespace(
        population_size=1,
        frame_counts=[3],
        identity={"role": "P", "role_manifest_sha256": "a" * 64},
        batch=unavailable,
    )
    receipt = training.fit_producer(source, recipe, tmp_path, device="cpu")
    assert receipt["status"] == "PAUSED_RESOURCE" and receipt["complete_state"]
    assert receipt["completed_updates"] == receipt["recovery_upper"] == 0
    directory = tmp_path / "fits/E_A5_MATCHED"
    pointer = read_json_shared(directory / "CHECKPOINT_POINTER.json")
    assert pointer["checkpoint_sha256"] == sha256_file(directory / "checkpoint_last.pt")
    payload = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    for key, value in before.items():
        torch.testing.assert_close(payload["model_state_dict"][key], value, rtol=0, atol=0)
    assert payload["cursor"]["position"] == 0 and payload["accumulation_index"] == 0
    assert not payload["optimizer_state_dict"]["state"]
    assert "scheduler_state_dict" in payload and "sampler_generator_state" in payload
    assert "torch_rng_state" in payload and "numpy_random_state" in payload
    model.weight.data.zero_()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
    restored = training.ProducerCheckpoint(directory, payload["identity"], recipe.updates)
    cursor = restored.restore(model, optimizer, scheduler, torch.Generator())
    assert restored.completed == 0 and cursor["position"] == 0
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fit,module",
    [
        ("E_A5_MATCHED", "operational.rgb_port_concurrent.producer"),
        ("E_C2F_MATCHED", "operational.rgb_port_c2f_graph.producer"),
        ("R_A5", "operational.rgb_port_acceleration.producer"),
    ],
)
def test_routing_retains_admitted_graph_delegate_and_every_original_argument(fit, module):
    command = ["python", "-m", module, "--graph-freeze", "original.json", "--", "--fit-id", fit]
    routed = queue.route(command, Path("io.json"))
    assert routed[2] == queue.PRODUCER_MODULE
    assert routed[routed.index("--delegate-module") + 1] == module
    assert routed[routed.index("--") + 1 :] == command[3:]
    assert queue.route(["python", "-m", "operational.rgb_port.train_heads"], Path("io.json")) == [
        "python",
        "-m",
        "operational.rgb_port.train_heads",
    ]


def test_only_authorized_event_pair_is_detached():
    process = SimpleNamespace(pid=12, wait=lambda: 0)
    launch = queue.detach_factory(lambda *_a, **_k: process)
    for fit in ("E_A5_MATCHED", "E_C2F_MATCHED"):
        child = launch(["python", "-m", queue.PRODUCER_MODULE, "--fit-id", fit])
        with pytest.raises(queue.concurrent_queue._DetachedChildError):
            child.wait()
    assert launch(["python", "-m", queue.PRODUCER_MODULE, "--fit-id", "R_A5"]) is process


def test_supervisor_retries_disk_io_without_hiding_contract_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "validate_freeze", lambda _: {})
    monkeypatch.setattr(queue, "installed", lambda _: nullcontext())
    sleeps = []
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)
    calls = 0

    def resume(_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EINVAL, "external disk disappeared")
        return 0

    monkeypatch.setattr(queue.frozen_queue, "main", resume)
    assert queue.main(["resume", "--run", str(tmp_path)]) == 0
    assert calls == 2 and sleeps == [30]
    assert not (tmp_path / "IO_RECOVERY_OWNER.json").exists()
    assert (
        read_json_shared(tmp_path / "io_recovery/SUPERVISOR_DISK_PAUSE.json")["status"]
        == "WAITING_DISK_IO"
    )
    monkeypatch.setattr(
        queue.frozen_queue, "main", lambda _: (_ for _ in ()).throw(ValueError("bad freeze"))
    )
    with pytest.raises(ValueError, match="bad freeze"):
        queue.main(["resume", "--run", str(tmp_path)])


def test_existing_live_supervisor_is_never_duplicated(tmp_path, monkeypatch):
    import os

    import psutil

    monkeypatch.setattr(queue, "validate_freeze", lambda _: {})
    owner = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
    atomic_write_json(tmp_path / "OWNER.json", owner)
    monkeypatch.setattr(
        queue.frozen_queue,
        "main",
        lambda _: (_ for _ in ()).throw(AssertionError("Duplicate launch")),
    )
    with pytest.raises(RuntimeError, match="refusing duplicate"):
        queue.main(["resume", "--run", str(tmp_path)])
    assert read_json_shared(tmp_path / "OWNER.json") == owner
    assert not (tmp_path / "IO_RECOVERY_OWNER.json").exists()


def test_only_native_reconciled_orphans_are_expedited():
    state = {
        "tasks": {
            "orphan": {
                "status": "PAUSED_RESOURCE",
                "retry_after_unix": 900,
                "reason": "orphaned child reconciled for durable resume",
            },
            "memory": {
                "status": "PAUSED_RESOURCE",
                "retry_after_unix": 900,
                "reason": "child reported a recoverable resource pause",
            },
            "live": {"status": "RUNNING", "retry_after_unix": 900},
        }
    }
    assert queue.expedite_reconciled_orphans(state, {"orphan", "memory", "live"})
    assert state["tasks"]["orphan"]["retry_after_unix"] == 0
    assert state["tasks"]["memory"]["retry_after_unix"] == 900
    assert state["tasks"]["live"]["retry_after_unix"] == 900


def test_overlay_freeze_rejects_source_changes(tmp_path, monkeypatch):
    from operational.rgb_port.recipe import canonical_sha256

    monkeypatch.setattr(common, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    qa = tmp_path / "qa.xml"
    qa.write_text("passed", encoding="utf-8")
    value = {
        "schema": "rgb_port_io_recovery_freeze_v1",
        "scientific_changes": False,
        "optimizer_updates_for_qa": 0,
        "files": {"source.py": sha256_file(source)},
        "upstream": {},
        "qa": {"path": str(qa), "sha256": sha256_file(qa)},
    }
    value["identity_sha256"] = canonical_sha256(value)
    atomic_write_json(tmp_path / common.FREEZE_NAME, value)
    assert common.validate_freeze(tmp_path) == value
    source.write_text("unreviewed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        common.validate_freeze(tmp_path)
