"""Zero-optimizer tests for the additive RGB-PORT producer acceleration."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_acceleration.producer import (
    ForwardDispatcher,
    LazyCPUComponents,
    _same,
)


def test_dispatcher_compiles_only_exact_b32_t3() -> None:
    calls: list[str] = []

    def eager(inputs, delta, *, return_dense_features=False):
        del delta, return_dense_features
        calls.append("eager")
        return inputs.sum()

    def compiled(inputs, delta, *, return_dense_features=False):
        del delta, return_dense_features
        calls.append("compiled")
        return inputs.sum()

    dispatcher = ForwardDispatcher(eager, compiled)
    dispatcher(torch.ones(32, 3, 1), torch.ones(32, 2), return_dense_features=True)
    dispatcher(torch.ones(31, 3, 1), torch.ones(31, 2), return_dense_features=True)
    dispatcher(torch.ones(32, 2, 1), torch.ones(32, 1), return_dense_features=True)
    assert calls == ["compiled", "eager", "eager"]
    assert dispatcher.snapshot() == {"compiled_calls": 1, "eager_calls": 2}


def test_batched_components_match_original_fp32_cpu_scalars_and_preserve_rng() -> None:
    torch.manual_seed(91)
    np.random.seed(92)
    random.seed(93)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    values = {
        "fp32": torch.tensor(1.25, dtype=torch.float32),
        "bf16": torch.tensor(-2.5, dtype=torch.bfloat16),
        "fp64": torch.tensor(3.125, dtype=torch.float64),
    }
    expected = {name: value.detach().float().cpu().clone() for name, value in values.items()}
    counter = {"component_cpu_transfers": 0}
    lazy = LazyCPUComponents(values, counter)
    assert tuple(lazy) == tuple(values) and len(lazy) == 3 and bool(lazy)
    assert tuple(dict(lazy)) == tuple(values)
    observed = dict(lazy.items())
    assert tuple(observed) == tuple(values)
    assert all(torch.equal(observed[name], expected[name]) for name in expected)
    assert counter == {"component_cpu_transfers": 1}
    dict(lazy.items())
    assert counter == {"component_cpu_transfers": 1}
    assert torch.equal(torch_before, torch.get_rng_state())
    assert _same(numpy_before, np.random.get_state())
    assert _same(python_before, random.getstate())


def test_recursive_exact_state_comparison_handles_tensors_arrays_and_order() -> None:
    value = {
        "tensor": torch.tensor([1, 2]),
        "array": np.asarray([3.0, 4.0], np.float32),
        "nested": ("x", [5, 6]),
    }
    clone = {
        "tensor": value["tensor"].clone(),
        "array": value["array"].copy(),
        "nested": ("x", [5, 6]),
    }
    assert _same(value, clone)
    assert not _same(value, {**clone, "tensor": torch.tensor([1, 3])})
    assert not _same(value, dict(reversed(tuple(clone.items()))))


def test_pending_execution_proof_is_written_before_native_save(tmp_path) -> None:
    from operational.rgb_port_acceleration.producer import Runtime

    runtime = object.__new__(Runtime)
    runtime.fit_id = "R_A5"
    runtime.graph_enabled = False
    runtime.freeze_sha256 = "f" * 64
    runtime.freeze_file_sha256 = "p" * 64
    runtime.freeze = {
        "admission_sha256": "a" * 64,
        "source_sha256": {"source": "s" * 64},
        "torch_backend_source_sha256": {"backend": "b" * 64},
    }
    runtime.receipt_dir = tmp_path / "acceleration_checkpoints"
    runtime.pending_path = runtime.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"

    class State:
        durable = 100
        completed = 125
        identity_sha256 = "i" * 64

    runtime.before_save(State())
    pending = read_json_shared(runtime.pending_path)
    assert pending["schema"] == "rgb_port_acceleration_pending_v1"
    assert pending["start_update"] == 100 and pending["end_update"] == 125
    assert pending["checkpoint_version"] == "checkpoint_000125.pt"
    assert pending["mode"] == "batched_metrics_only"


def _runtime(tmp_path, *, fit_id: str = "R_A5", graph: bool = False):
    from operational.rgb_port_acceleration.producer import Runtime

    runtime = object.__new__(Runtime)
    runtime.fit_id = fit_id
    runtime.graph_enabled = graph
    runtime.freeze_sha256 = "f" * 64
    runtime.freeze_file_sha256 = "p" * 64
    runtime.freeze = {
        "origin": {
            "completed_updates": 6189,
            "checkpoint_sha256": "o" * 64,
            "identity_sha256": "fit-identity",
        },
        "admission_sha256": "a" * 64,
        "source_sha256": {"source": "s" * 64},
        "torch_backend_source_sha256": {"backend": "b" * 64},
    }
    fit = tmp_path / "fits" / fit_id
    runtime.runtime_path = fit / "ACCELERATION_RUNTIME.json"
    runtime.prewarm_path = fit / "PREWARM_QA.json"
    runtime.receipt_dir = fit / "acceleration_checkpoints"
    runtime.pending_path = runtime.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"
    runtime.dispatcher = None
    runtime.counters = {"component_cpu_transfers": 0, "warmup_backward_passes": 0}
    return runtime


def _state(tmp_path, *, completed: int = 125):
    fit = tmp_path / "fits/R_A5"
    version = fit / "checkpoint_versions" / f"checkpoint_{completed:06d}.pt"
    version.parent.mkdir(parents=True, exist_ok=True)
    version.write_bytes(f"checkpoint-{completed}".encode())
    pointer = fit / "CHECKPOINT_POINTER.json"
    atomic_write_json(
        pointer,
        {
            "version": version.name,
            "completed_updates": completed,
            "checkpoint_sha256": sha256_file(version),
            "identity_sha256": "fit-identity",
        },
    )
    return SimpleNamespace(
        durable=completed - 25,
        completed=completed,
        identity_sha256="fit-identity",
        path=version,
        pointer=pointer,
    )


def test_restart_does_not_invalidate_immutable_checkpoint_runtime(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    state = _state(tmp_path)
    runtime._ready(100, "old-checkpoint", {"initial_session": True})
    runtime.before_save(state)
    runtime.after_save(state)

    receipt_path = runtime.receipt_dir / "checkpoint_000125.json"
    receipt = read_json_shared(receipt_path)
    archive = runtime.receipt_dir / receipt["runtime_snapshot"]
    archive_sha = sha256_file(archive)
    assert archive.name == "runtime_000125.json"
    assert receipt["runtime_receipt_sha256"] == archive_sha

    # A later process may replace mutable session state before its next save.
    runtime._ready(125, sha256_file(state.path), {"restarted_session": True})
    assert sha256_file(runtime.runtime_path) != archive_sha
    runtime.verify_or_repair_lineage(state)
    assert sha256_file(archive) == archive_sha
    assert read_json_shared(receipt_path) == receipt


def test_pending_repair_syncs_runtime_and_seals_snapshot_before_receipt(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    state = _state(tmp_path)
    runtime._ready(100, "stale-checkpoint", {"prior_session": True})
    runtime.before_save(state)

    runtime.verify_or_repair_lineage(state)

    repaired_runtime = read_json_shared(runtime.runtime_path)
    receipt = read_json_shared(runtime.receipt_dir / "checkpoint_000125.json")
    archive = runtime.receipt_dir / receipt["runtime_snapshot"]
    assert repaired_runtime["last_saved_update"] == 125
    assert repaired_runtime["last_checkpoint_sha256"] == sha256_file(state.path)
    assert receipt["status"] == "RECOVERED_FROM_PENDING"
    assert receipt["runtime_receipt_sha256"] == sha256_file(archive)
    assert read_json_shared(archive)["last_checkpoint_sha256"] == sha256_file(state.path)
    assert not runtime.pending_path.exists()


def test_native_origin_shortcut_is_graph_only(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    state = _state(tmp_path, completed=6189)
    with pytest.raises(RuntimeError, match="lacks receipt"):
        runtime.verify_or_repair_lineage(state)
