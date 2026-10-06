"""CPU-only contracts for the optional TRAIN40 OpenMP idle-wait experiment."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.efficient_context.common import ROOT, digest
from operational.train40_system import (
    controller_idle_wait,
    engine_idle_wait,
    idle_wait_freeze,
)
from operational.train40_system.durable_io import atomic_json


def test_engine_entry_configures_only_admitted_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNCHANGED_SENTINEL", "same")
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("KMP_BLOCKTIME", "200")
    before = dict(os.environ)
    engine_idle_wait.configure_environment()
    changed = {name for name in os.environ if os.environ.get(name) != before.get(name)}
    assert changed == {"OMP_WAIT_POLICY", "KMP_BLOCKTIME"}
    assert engine_idle_wait.IDLE_WAIT_ENVIRONMENT == {
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
    }
    assert os.environ["UNCHANGED_SENTINEL"] == "same"


def test_engine_source_defers_torch_dependent_imports_until_after_configuration() -> None:
    path = ROOT / "operational/train40_system/engine_idle_wait.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) for node in top_imports)
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    first_statement = run.body[1] if isinstance(run.body[0], ast.Expr) else run.body[0]
    assert ast.unparse(first_statement) == "configure_environment()"


def test_controller_routes_only_exact_encoder_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, list[str]]] = []

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> SimpleNamespace:
        assert output == tmp_path
        calls.append((tag, module, arguments))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(controller_idle_wait, "_frozen_launch", launch)
    controller_idle_wait.launch(
        tmp_path, "a5", "operational.train40_system.engine", ["--arm", "a5"]
    )
    controller_idle_wait.launch(tmp_path, "teacher", "frozen.other", ["x"])
    assert calls == [
        ("a5", "operational.train40_system.engine_idle_wait", ["--arm", "a5"]),
        ("teacher", "frozen.other", ["x"]),
    ]
    with pytest.raises(ValueError, match="engine task"):
        controller_idle_wait.launch(
            tmp_path, "a5", "operational.train40_system.engine", ["--arm", "c2f"]
        )


def test_controller_composes_and_restores_frozen_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    atomic_json(tmp_path / "IDLE_WAIT_FREEZE.json", {"files": []})
    frozen = controller_idle_wait.controller_overlap.launch
    observed: list[object] = []

    def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
        observed.extend(
            [output, raw_root, teacher_path, controller_idle_wait.controller_overlap.launch]
        )

    monkeypatch.setattr(controller_idle_wait.controller_pause_safe, "run", run)
    raw, teacher = tmp_path / "raw", tmp_path / "teacher"
    controller_idle_wait.run(tmp_path, raw, teacher)
    assert observed[:3] == [tmp_path, raw, teacher]
    assert observed[3] is controller_idle_wait.launch
    assert controller_idle_wait.controller_overlap.launch is frozen


def test_cpu_benchmark_source_contains_no_cuda_or_optimizer_calls() -> None:
    source = (ROOT / "operational/train40_system/idle_wait.py").read_text(encoding="utf-8")
    assert "torch.cuda" not in source
    assert "optimizer.step" not in source
    assert "torch.set_num_threads(4)" in source
    assert "time.sleep(SLEEP_SECONDS)" in source
    assert "SLEEP_SECONDS = 0.100" in source
    assert "PROFILED_GPU_ARITHMETIC_SECONDS = 0.1093" in source
    assert "(3, 10, 64, 64)" in source


def _benchmark(mode: str, cpu: float, wall: float) -> dict:
    return {
        "schema": "train40_idle_wait_cpu_benchmark_v1",
        "status": "COMPLETE",
        "mode": mode,
        "iterations": 30,
        "sleep_seconds_per_iteration": 0.100,
        "calibration": {
            "basis": "measured full-step GPU active-union arithmetic",
            "measured_seconds": 0.1093,
            "benchmark_sleep_seconds": 0.100,
            "earlier_20ms_pilot_was_not_representative": True,
        },
        "batch_shapes": [[32, 3, 10, 64, 64], [32, 3, 4]],
        "tensor_sha256": "same-tensor",
        "process_cpu_seconds": cpu,
        "wall_seconds": wall,
        "environment": {
            "OMP_WAIT_POLICY": "PASSIVE" if mode == "passive" else None,
            "KMP_BLOCKTIME": "0" if mode == "passive" else None,
            "OMP_NUM_THREADS": "4",
        },
        "torch_num_threads": 4,
        "kmp_blocktime_after": {"available": mode == "passive", "value": 0},
        "cuda_apis_called": False,
        "optimizer_updates": 0,
    }


def _pilot_artifacts(output: Path) -> None:
    baseline = _benchmark("baseline", 3.03125, 0.9519994)
    passive = _benchmark("passive", 0.875, 1.1301643)
    for benchmark in (baseline, passive):
        benchmark["sleep_seconds_per_iteration"] = 0.020
        benchmark["tensor_sha256"] = "afd3d-pilot"
    atomic_json(output / "IDLE_WAIT_20MS_BASELINE.json", baseline)
    atomic_json(output / "IDLE_WAIT_20MS_PASSIVE.json", passive)
    for name in (
        "IDLE_WAIT_20MS_PYTEST.txt",
        "IDLE_WAIT_20MS_RUFF.txt",
        "IDLE_WAIT_20MS_PYRIGHT.txt",
    ):
        (output / name).write_text("preserved", encoding="utf-8")


def test_freeze_requires_cpu_reduction_equal_tensors_and_no_wall_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, marker in idle_wait_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    atomic_json(tmp_path / "COORDINATION_FREEZE.json", {"files": []})
    atomic_json(tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json", {"files": []})
    atomic_json(tmp_path / "IDLE_WAIT_BASELINE.json", _benchmark("baseline", 4.0, 3.0))
    atomic_json(tmp_path / "IDLE_WAIT_PASSIVE.json", _benchmark("passive", 1.0, 3.1))
    _pilot_artifacts(tmp_path)
    monkeypatch.setattr(idle_wait_freeze, "dependency_files", lambda sources: [])
    idle_wait_freeze.run(tmp_path)
    frozen = idle_wait_freeze.read(tmp_path / "IDLE_WAIT_FREEZE.json")
    assert frozen["process_cpu_ratio_passive_over_baseline"] == 0.25
    assert not frozen["training_throughput_acceleration_claimed"]
    assert frozen["benchmarks"]["baseline"]["sha256"] == digest(
        tmp_path / "IDLE_WAIT_BASELINE.json"
    )


@pytest.mark.parametrize(
    ("passive_cpu", "passive_wall", "tensor", "message"),
    [
        (4.0, 3.0, "same-tensor", "CPU cost"),
        (1.0, 4.0, "same-tensor", "wall time"),
        (1.0, 3.0, "different", "different CPU tensors"),
    ],
)
def test_freeze_rejects_unhelpful_or_inequivalent_measurement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    passive_cpu: float,
    passive_wall: float,
    tensor: str,
    message: str,
) -> None:
    for name, marker in idle_wait_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    atomic_json(tmp_path / "COORDINATION_FREEZE.json", {"files": []})
    atomic_json(tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json", {"files": []})
    atomic_json(tmp_path / "IDLE_WAIT_BASELINE.json", _benchmark("baseline", 4.0, 3.0))
    passive = _benchmark("passive", passive_cpu, passive_wall)
    passive["tensor_sha256"] = tensor
    atomic_json(tmp_path / "IDLE_WAIT_PASSIVE.json", passive)
    _pilot_artifacts(tmp_path)
    monkeypatch.setattr(idle_wait_freeze, "dependency_files", lambda sources: [])
    with pytest.raises(ValueError, match=message):
        idle_wait_freeze.run(tmp_path)
