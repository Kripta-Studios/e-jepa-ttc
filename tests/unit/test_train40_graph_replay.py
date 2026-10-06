"""CPU-only integration gates for admitted A5 graph replay."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.efficient_context.common import digest
from operational.train40_system import (
    controller_graph_replay,
    engine_graph_replay,
    graph_replay_freeze,
)
from operational.train40_system import (
    freeze as freeze_module,
)
from operational.train40_system.durable_io import atomic_json


class _Closable:
    def __init__(self, *, failure: bool = False) -> None:
        self.closed = False
        self.failure = failure

    def close(self) -> None:
        self.closed = True
        if self.failure:
            raise RuntimeError("close failure")


def test_cleanup_restores_bindings_and_closes_resources_when_restore_raises() -> None:
    original_forward = object()
    original_diagnostic = object()
    model = SimpleNamespace(forward=object())
    training = SimpleNamespace(_record_relational_fg_bg_diagnostic=object())
    data, prefetch = _Closable(), _Closable()

    def failed_restore() -> None:
        raise RuntimeError("warmup restore failure")

    with pytest.raises(RuntimeError, match="warmup restore"):
        engine_graph_replay._cleanup_runtime(
            model=model,
            original_forward=original_forward,
            training=training,
            original_diagnostic=original_diagnostic,
            data=data,
            prefetch=prefetch,
            restore_state=failed_restore,
        )
    assert model.forward is original_forward
    assert training._record_relational_fg_bg_diagnostic is original_diagnostic
    assert data.closed and prefetch.closed

    data, prefetch = _Closable(), _Closable(failure=True)
    with pytest.raises(RuntimeError, match="close failure"):
        engine_graph_replay._cleanup_runtime(
            model=model,
            original_forward=original_forward,
            training=training,
            original_diagnostic=original_diagnostic,
            data=data,
            prefetch=prefetch,
            restore_state=None,
        )
    assert data.closed and prefetch.closed


def test_graph_engine_is_a5_only_and_warmup_precedes_prefetch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only for A5"):
        engine_graph_replay.run(tmp_path, "c2f")

    source = Path(engine_graph_replay.__file__).read_text(encoding="utf-8")
    warm = source.index("for _ in range(3):")
    restore = source.index("cursor = state.restore", warm)
    prefetch = source.index("prefetch = DevicePrefetch", restore)
    first_science_update = source.index("state.begin_update()", prefetch)
    assert warm < restore < prefetch < first_science_update
    assert "optimizer.step()" not in source[warm:prefetch]
    assert source[warm:restore].count("optimizer.zero_grad(set_to_none=True)") == 1
    assert source.index("permitted, prewarm_telemetry = resource_guard(output)") < source.index(
        "_compile_forward(original_forward)"
    )
    assert source.count("optimizer.step()") == 1
    assert "requires an existing checkpoint; use engine_fast_scan_safe" in source
    assert "until the warmup epochs complete" in source


def test_scientific_loop_and_checkpoint_contract_preserve_overlap_ast() -> None:
    overlap_path = Path(engine_graph_replay.__file__).with_name("engine_overlap.py")
    overlap = ast.parse(overlap_path.read_text(encoding="utf-8"))
    graph = ast.parse(Path(engine_graph_replay.__file__).read_text(encoding="utf-8"))

    def run_node(tree: ast.Module) -> ast.FunctionDef:
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )

    def named_assignment(function: ast.FunctionDef, name: str) -> ast.Assign:
        return next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        )

    overlap_run, graph_run = run_node(overlap), run_node(graph)
    assert ast.dump(named_assignment(graph_run, "contract")) == ast.dump(
        named_assignment(overlap_run, "contract")
    )
    overlap_while = next(node for node in ast.walk(overlap_run) if isinstance(node, ast.While))
    graph_while = next(node for node in ast.walk(graph_run) if isinstance(node, ast.While))

    class RemoveRecompileGuard(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):  # noqa: N802
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            return None if names & {"live_compile_counters", "live_unique_graphs"} else node

        def visit_If(self, node: ast.If):  # noqa: N802
            if "live_unique_graphs" in ast.unparse(node.test):
                return None
            return self.generic_visit(node)

    cleaned = RemoveRecompileGuard().visit(graph_while)
    assert ast.dump(cleaned) == ast.dump(overlap_while)


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("a5", "operational.train40_system.engine_graph_replay"),
        ("c2f", "operational.train40_system.engine_fast_scan_safe"),
    ],
)
def test_controller_routes_only_exact_original_producers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str, expected: str
) -> None:
    calls = []
    monkeypatch.setattr(
        controller_graph_replay.controller_overlap,
        "_launch",
        lambda *arguments: calls.append(arguments),
    )
    controller_graph_replay.launch(
        tmp_path, arm, "operational.train40_system.engine", ["--arm", arm]
    )
    assert calls == [(tmp_path, arm, expected, ["--arm", arm])]


def _freeze_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "root"
    source_dir = root / "operational/train40_system"
    test_dir = root / "tests/unit"
    source_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    monkeypatch.setattr(graph_replay_freeze, "ROOT", root)
    monkeypatch.setattr(freeze_module, "ROOT", root)
    output = tmp_path / "output"
    (output / "graph_launch_admission").mkdir(parents=True)
    for name in (
        "COORDINATION_FREEZE.json",
        "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "ADAPTIVE_CACHE_FREEZE.json",
        "FAST_PROCESS_SCAN_FREEZE.json",
    ):
        atomic_json(output / name, {"files": []})
    for name, marker in graph_replay_freeze.QA_CHECKS.items():
        (output / name).write_text(marker, encoding="utf-8")
    for name in (
        "engine_graph_replay.py",
        "controller_graph_replay.py",
        "graph_replay_freeze.py",
        "graph_launch_admission.py",
        "engine_fast_scan_safe.py",
    ):
        (source_dir / name).write_text(name, encoding="utf-8")
    for name in ("test_train40_graph_replay.py", "test_train40_fast_scan_safe.py"):
        (test_dir / name).write_text(name, encoding="utf-8")
    admission_source = source_dir / "graph_launch_admission.py"
    required = {
        name: True
        for name in (
            "exact_loss_and_all_components",
            "exact_all_model_output_tensors",
            "all_gradient_keys_shapes_and_finite_audited",
            "gradient_parity_within_baseline_repeatability",
            "repeated_dropout_CPU_and_CUDA_RNG_trajectory_exact",
            "B32_B8_B32_output_loss_components_and_RNG_trajectory_exact",
            "batch8_uses_original_eager_forward",
            "model_parameter_identity_and_state_dict_keys_unchanged",
            "original_shape_and_delta_t_guards_retained",
        )
    }
    atomic_json(
        output / "graph_launch_admission/REAL_GRAPH_LAUNCH_ADMISSION.json",
        {
            "status": "PASSED",
            "compatible": True,
            "arm": "a5",
            **required,
            "gradient_tensors_compared": 59,
            "optimizer_updates": 0,
            "training_checkpoint_modified": False,
            "steady_recompiles": 0,
            "backend": "cudagraphs",
            "source_sha256": digest(admission_source),
            "checkpoint_sha256": "checkpoint",
            "checkpoint_updates": 17469,
            "backend_source_sha256": "backend",
            "repeated_gradient_admission": {},
            "steady_eager_seconds": [0.17, 0.16, 0.17],
            "steady_compiled_seconds": [0.56, 0.13, 0.12],
            "first_compiled_forward_backward_seconds": 11.0,
            "launch_profiles": {"compiled": {"cudaGraphLaunch": {"count": 10}}},
            "peak_reserved_vram_bytes": {"compiled_replay": 1},
        },
    )
    return output


def test_freeze_requires_exact_real_admission_and_binds_safe_c2f(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _freeze_fixture(monkeypatch, tmp_path)
    graph_replay_freeze.run(output)
    frozen = graph_replay_freeze.read(output / "GRAPH_REPLAY_FREEZE.json")
    assert frozen["warmup_optimizer_updates"] == 0
    assert frozen["a5_only_graph_replay"] is True
    assert frozen["fast_scan_safe_engine_sha256"] == digest(
        tmp_path / "root/operational/train40_system/engine_fast_scan_safe.py"
    )

    (output / "GRAPH_REPLAY_FREEZE.json").unlink()
    admission_path = output / "graph_launch_admission/REAL_GRAPH_LAUNCH_ADMISSION.json"
    admission = graph_replay_freeze.read(admission_path)
    admission["B32_B8_B32_output_loss_components_and_RNG_trajectory_exact"] = False
    atomic_json(admission_path, admission)
    with pytest.raises(ValueError, match="admission must pass"):
        graph_replay_freeze.run(output)
