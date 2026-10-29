"""CPU, zero-optimizer tests for deterministic RGB-PORT Pipeline V2 prefetch."""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_pipeline_v2.prefetch import DepthOneEventPrefetch


class FakeSource:
    def __init__(self, *, population: int = 10) -> None:
        self.population_size = population
        self.frame_counts = [3 if index % 2 == 0 else 2 for index in range(population)]
        self.input_span_us = [300_000] * population
        self.identity = {"role": "P"}
        self.calls: list[tuple[tuple[int, ...], str]] = []
        self.fail: OSError | None = None

    def batch(self, ids: list[int], modality: str) -> dict[str, Any]:
        self.calls.append((tuple(ids), modality))
        if self.fail is not None:
            raise self.fail
        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "features": torch.tensor(ids, dtype=torch.float32).unsqueeze(1),
            "metadata": [{"sample_token": f"sample-{index}"} for index in ids],
        }


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_prefetch_matches_exact_batches_and_preserves_rng_cursor() -> None:
    source = FakeSource()
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(10)}
    cursor_before = {"epoch": 1, "position": 0, "order": cursor["order"].clone()}
    torch.manual_seed(31)
    np.random.seed(32)
    random.seed(33)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    wrapper = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    wrapper.limit_effective_batches(2)
    wrapper.bind(cursor)

    for ids in ([1, 3], [0, 2], [5, 7], [4, 6]):
        expected = {
            "ids": torch.tensor(ids),
            "features": torch.tensor(ids, dtype=torch.float32).unsqueeze(1),
            "metadata": [{"sample_token": f"sample-{index}"} for index in ids],
        }
        assert _equal(wrapper.batch(ids, "event"), expected)
    wrapper.bind(cursor)
    wrapper.close()

    assert source.calls[:4] == [
        ((1, 3), "event"),
        ((0, 2), "event"),
        ((5, 7), "event"),
        ((4, 6), "event"),
    ]
    assert source.calls[4:] == []
    assert cursor["epoch"] == cursor_before["epoch"]
    assert cursor["position"] == cursor_before["position"]
    assert torch.equal(cursor["order"], cursor_before["order"])
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert repr(numpy_rng) == repr(np.random.get_state())
    assert python_rng == random.getstate()
    timing = wrapper.timing_snapshot()
    assert timing["source_batches"] == len(source.calls)
    assert timing["prefetch_hits"] == 4
    assert timing["submitted"] == timing["consumed"] == 4
    assert timing["outstanding_max"] == 1 and timing["outstanding_current"] == 0


def test_tail_then_natural_epoch_order_change_is_detected() -> None:
    source = FakeSource(population=6)
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(6)}
    wrapper = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    wrapper.bind(cursor)
    wrapper.batch([1, 3], "event")
    wrapper.batch([0, 2], "event")
    wrapper.batch([5], "event")
    wrapper.batch([4], "event")

    cursor.update(epoch=2, position=0, order=torch.tensor([5, 4, 3, 2, 1, 0]))
    assert wrapper.batch([5, 3], "event")["metadata"] == [
        {"sample_token": "sample-5"},
        {"sample_token": "sample-3"},
    ]
    wrapper.close()
    assert wrapper.timing_snapshot()["synchronous_boundaries"] == 1


def test_oserror_is_deferred_to_requested_boundary() -> None:
    source = FakeSource(population=4)
    source.fail = OSError(5, "transient read")
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(4)}
    wrapper = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    wrapper.bind(cursor)
    with pytest.raises(OSError, match="transient read"):
        wrapper.batch([1, 3], "event")
    assert cursor["position"] == 0
    wrapper.close()


def test_close_cancels_or_joins_single_future_without_cursor_advance() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingSource(FakeSource):
        def batch(self, ids: list[int], modality: str) -> dict[str, Any]:
            started.set()
            assert release.wait(timeout=2)
            return super().batch(ids, modality)

    source = BlockingSource(population=4)
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(4)}
    wrapper = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    wrapper.bind(cursor)
    assert started.wait(timeout=1)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (wrapper.close(), closed.set()))
    closer.start()
    time.sleep(0.02)
    assert not closed.is_set()
    release.set()
    closer.join(timeout=2)
    assert closed.is_set() and cursor["position"] == 0


def test_mismatched_request_fails_without_consuming_hidden_batch() -> None:
    source = FakeSource(population=4)
    cursor = {"epoch": 1, "position": 0, "order": torch.arange(4)}
    wrapper = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    wrapper.bind(cursor)
    with pytest.raises(RuntimeError, match="differs from canonical cursor"):
        wrapper.batch([0, 2], "event")
    assert cursor["position"] == 0
    wrapper.close()


def test_lineage_only_runtime_does_not_patch_training_hot_path() -> None:
    import operational.rgb_port.train_producers as training
    from operational.rgb_port_pipeline_v2.producer import Runtime, installed

    runtime = object.__new__(Runtime)
    runtime.prefetch_enabled = False
    runtime.runtime_mode = "ROLLED_BACK_V1"
    originals = {
        "restore": training.ProducerCheckpoint.restore,
        "save": training.ProducerCheckpoint.save,
        "to_device": training._to_device,
        "loss": training._producer_loss,
        "fast": training._fast_resource_guard,
        "full": training._resource_guard,
    }
    with installed(runtime):
        assert training.ProducerCheckpoint.restore is not originals["restore"]
        assert training.ProducerCheckpoint.save is not originals["save"]
        assert training._to_device is originals["to_device"]
        assert training._producer_loss is originals["loss"]
        assert training._fast_resource_guard is originals["fast"]
        assert training._resource_guard is originals["full"]
    assert training.ProducerCheckpoint.restore is originals["restore"]
    assert training.ProducerCheckpoint.save is originals["save"]


def _lineage_runtime(tmp_path: Path):
    from operational.rgb_port_pipeline_v2.producer import Runtime

    runtime = object.__new__(Runtime)
    runtime.fit_id = "E_A5_MATCHED"
    runtime.prefetch_enabled = False
    runtime.runtime_mode = "ROLLED_BACK_V1"
    runtime.freeze_sha256 = "f" * 64
    runtime.freeze_file_sha256 = "p" * 64
    runtime.v1_freeze_sha256 = "v" * 64
    runtime.v1_freeze_identity = "i" * 64
    runtime.freeze = {
        "identity_sha256": runtime.freeze_sha256,
        "source_sha256": {"producer": "s" * 64},
        "origins": {
            "E_A5_MATCHED": {
                "mode": "PAUSED_FULL_CHECKPOINT",
                "completed_updates": 100,
                "checkpoint_sha256": "unused",
                "identity_sha256": "fit-identity",
            }
        },
        "canary": {"scheduled_updates": 300},
    }
    runtime.fit = tmp_path / "fits/E_A5_MATCHED"
    runtime.runtime_path = runtime.fit / "PIPELINE_RUNTIME.json"
    runtime.receipt_dir = runtime.fit / "pipeline_checkpoints"
    runtime.pending_path = runtime.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"
    runtime.timings = {}
    runtime.science_started = time.perf_counter()
    runtime.canary_marks = []
    runtime._last_snapshot_total = 0.0
    runtime._last_snapshot_timings = {}
    source = FakeSource(population=4)
    runtime.prefetch = DepthOneEventPrefetch(source, effective_batch_size=4, microbatch_size=2)
    version = runtime.fit / "checkpoint_versions/checkpoint_000125.pt"
    version.parent.mkdir(parents=True)
    version.write_bytes(b"checkpoint")
    pointer = runtime.fit / "CHECKPOINT_POINTER.json"
    atomic_write_json(
        pointer,
        {
            "version": version.name,
            "completed_updates": 125,
            "checkpoint_sha256": sha256_file(version),
            "identity_sha256": "fit-identity",
        },
    )
    state = SimpleNamespace(
        completed=125,
        durable=100,
        identity_sha256="fit-identity",
        path=version,
        pointer=pointer,
    )
    return runtime, state


def test_pending_repair_rejects_tampered_parent_binding(tmp_path) -> None:
    runtime, state = _lineage_runtime(tmp_path)
    runtime.before_save(state)
    pending = read_json_shared(runtime.pending_path)
    pending["original_acceleration_freeze_sha256"] = "tampered"
    atomic_write_json(runtime.pending_path, pending)
    with pytest.raises(RuntimeError, match="bindings differ"):
        runtime.after_restore(state, {})
    runtime.close()


def test_pending_repair_reuses_timing_written_before_receipt_crash(tmp_path) -> None:
    runtime, state = _lineage_runtime(tmp_path)
    runtime.before_save(state)
    timing_path = runtime.receipt_dir / "timings_000125.json"
    timing = runtime._timing_value(state)
    atomic_write_json(timing_path, timing)
    timing_sha = sha256_file(timing_path)

    runtime.science_started = time.perf_counter() - 99.0
    runtime.after_restore(state, {})

    receipt = read_json_shared(runtime.receipt_dir / "checkpoint_000125.json")
    assert receipt["status"] == "RECOVERED_FROM_PENDING"
    assert receipt["timings_sha256"] == timing_sha == sha256_file(timing_path)
    assert not runtime.pending_path.exists()
    runtime.close()
