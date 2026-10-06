"""CPU-only contracts for the additive C2F CUDA-graph production route."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from e_jepa_ttc.training import causal_scale_eap as training
from operational.efficient_context.common import digest
from operational.train40_system import c2f_graph_core
from operational.train40_system import c2f_graph_freeze as freeze
from operational.train40_system import engine_c2f_graph_replay as engine
from operational.train40_system import freeze as freeze_module
from operational.train40_system.durable_io import atomic_json


class _State(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scalar", torch.tensor(2.0))
        self.register_buffer("vector", torch.tensor([1, 3], dtype=torch.int64))


def test_scalar_safe_hash_supports_scalars_and_preserves_nonscalar_identity() -> None:
    module = _State()
    first = engine.scalar_safe_module_tensor_sha256(module)
    assert first == engine.scalar_safe_module_tensor_sha256(module)
    assert len(first) == 64
    nonscalar = nn.Sequential(nn.Linear(2, 3))
    assert engine.scalar_safe_module_tensor_sha256(nonscalar) == training._module_tensor_sha256(
        nonscalar
    )


def test_planned_prefetch_prepares_order_before_delegating() -> None:
    calls: list[tuple[str, Any]] = []

    class Base:
        def __init__(self) -> None:
            self.order = None
            self.pending = {}
            self.expected = 0
            self.scheduled = 0
            self.source = SimpleNamespace(
                prepare_order=lambda *args: calls.append(("prepare", args))
            )

        def take(self, order: torch.Tensor, start: int, batch_size: int) -> str:
            calls.append(("take", (order, start, batch_size)))
            return "ticket"

    prefetch = engine.planned_prefetch_type(Base)()
    order = torch.arange(8)
    assert prefetch.take(order, 2, 4) == "ticket"
    assert [name for name, _ in calls] == ["prepare", "take"]


def test_inventory_repair_is_bounded_and_keeps_other_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor = engine.C2FResourceMonitor.__new__(engine.C2FResourceMonitor)
    monitor.output = tmp_path
    monkeypatch.setattr(
        engine.psutil, "Process", lambda: SimpleNamespace(pid=7, create_time=lambda: 1.0)
    )
    clean = {
        "inventory_unreadable_python_pids": [],
        "other_heavy_processes": [],
        "tree_rss_bytes": 1,
        "host_available_bytes": 3 * 1024**3,
        "disk_free_bytes": 30_000_000_000,
        "processes_seen": 5,
        "python_cmdline_reads": 2,
    }
    monkeypatch.setattr(engine.resource_monitor, "_collect_sample", lambda *_: clean)
    allowed, telemetry = monitor._repair_inventory(
        False, {"reasons": ["RESOURCE_MONITOR_INVENTORY_INCOMPLETE"]}
    )
    assert allowed and telemetry["reasons"] == []
    allowed, telemetry = monitor._repair_inventory(
        False,
        {"reasons": ["RESOURCE_MONITOR_INVENTORY_INCOMPLETE", "PHYSICAL_UPDATE_CAP"]},
    )
    assert not allowed and telemetry["reasons"] == ["PHYSICAL_UPDATE_CAP"]


def test_wrapper_restores_every_binding_after_delegate_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Monitor:
        def __init__(self, _: Path) -> None:
            pass

        def start(self) -> None:
            pass

        def guard(self, _: Path) -> tuple[bool, dict[str, Any]]:
            return True, {}

        def snapshot(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(engine, "verify_admission", lambda _: {})
    monkeypatch.setattr(engine, "C2FResourceMonitor", Monitor)
    originals = (
        c2f_graph_core.SealedInputs,
        c2f_graph_core.DevicePrefetch,
        c2f_graph_core.resource_guard,
        c2f_graph_core.scan_snapshot,
        training._module_tensor_sha256,
    )
    monkeypatch.setattr(
        c2f_graph_core,
        "run",
        lambda *_: (_ for _ in ()).throw(RuntimeError("delegate failed")),
    )
    with pytest.raises(RuntimeError, match="delegate failed"):
        engine.run(tmp_path, "c2f")
    assert originals == (
        c2f_graph_core.SealedInputs,
        c2f_graph_core.DevicePrefetch,
        c2f_graph_core.resource_guard,
        c2f_graph_core.scan_snapshot,
        training._module_tensor_sha256,
    )


def test_freeze_requires_exact_c2f_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    source_dir = root / "operational/train40_system"
    tests = root / "tests/unit"
    source_dir.mkdir(parents=True)
    tests.mkdir(parents=True)
    monkeypatch.setattr(freeze, "ROOT", root)
    monkeypatch.setattr(freeze_module, "ROOT", root)
    output = tmp_path / "output"
    output.mkdir()
    for name, marker in freeze.QA_CHECKS.items():
        (output / name).write_text(marker, encoding="utf-8")
    for name in (
        "COORDINATION_FREEZE.json",
        "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "ADAPTIVE_CACHE_FREEZE.json",
        "FAST_PROCESS_SCAN_FREEZE.json",
        "HOST_FAST_4_FREEZE.json",
    ):
        atomic_json(output / name, {"files": []})
    source_names = (
        "c2f_graph_core.py",
        "engine_c2f_graph_replay.py",
        "controller_c2f_graph.py",
        "c2f_graph_freeze.py",
        "c2f_graph_admission.py",
        "process_inputs_fast_4.py",
        "resource_monitor.py",
    )
    for name in source_names:
        (source_dir / name).write_text(name, encoding="utf-8")
    for name in (
        "test_train40_c2f_graph_replay.py",
        "test_train40_c2f_graph_controller.py",
        "test_train40_c2f_graph_admission.py",
    ):
        (tests / name).write_text(name, encoding="utf-8")
    backend = tmp_path / "backend.py"
    backend.write_text("backend", encoding="utf-8")
    admission_dir = output / "c2f_graph_launch_admission"
    admission_dir.mkdir()
    admission = {
        "status": "PASSED",
        "compatible": True,
        "arm": "c2f",
        **{name: True for name in freeze.REQUIRED_PARITY},
        "gradient_tensors_compared": 63,
        "checkpoint_updates": 13817,
        "optimizer_updates": 0,
        "training_checkpoint_modified": False,
        "steady_recompiles": 0,
        "backend": "cudagraphs",
        "backend_source": str(backend),
        "backend_source_sha256": digest(backend),
        "source_sha256": digest(source_dir / "c2f_graph_admission.py"),
        "launch_profiles": {"compiled": {"GraphLaunch": 1}},
        "checkpoint_sha256": "a" * 64,
        "repeated_gradient_admission": {},
        "steady_eager_seconds": [1.0],
        "steady_compiled_seconds": [0.3],
        "peak_reserved_vram_bytes": {
            "eager_after_graph_capture": 3_000_000_000,
            "compiled_replay": 3_100_000_000,
        },
    }
    atomic_json(admission_dir / "REAL_GRAPH_LAUNCH_ADMISSION.json", admission)
    freeze.run(output)
    frozen = freeze.read(output / "C2F_GRAPH_REPLAY_FREEZE.json")
    assert frozen["gradient_parity_admission"] == {}
    assert frozen["b8_original_eager_fallback"] is True
