"""CPU-only contract checks for the admitted TRAIN40 overlap source variant."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.efficient_context.common import ROOT, digest
from operational.train40_system import controller_overlap, coordination_freeze
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine_overlap import SOURCE_REPLACEMENTS, pause_request

ORIGINAL = ROOT / "operational/train40_system/engine.py"
OVERLAP = ROOT / "operational/train40_system/engine_overlap.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _definition(tree: ast.Module, name: str) -> ast.ClassDef | ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )


def _named_assignment(function: ast.FunctionDef, name: str) -> ast.Assign:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


def _call(function: ast.FunctionDef, name: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    )


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def test_original_scientific_ast_is_retained_outside_explicit_coordination_replacements() -> None:
    """Reconstruct the immutable scientific spine from both source ASTs."""
    original, overlap = _tree(ORIGINAL), _tree(OVERLAP)
    assert _dump(_definition(original, "SealedInputs")) == _dump(
        _definition(overlap, "SealedInputs")
    )
    original_run = _definition(original, "run")
    overlap_run = _definition(overlap, "run")
    assert isinstance(original_run, ast.FunctionDef)
    assert isinstance(overlap_run, ast.FunctionDef)
    for assignment in ("effective_training", "config", "loss_config", "contract", "cursor"):
        assert _dump(_named_assignment(original_run, assignment).value) == _dump(
            _named_assignment(overlap_run, assignment).value
        )
    for call in ("CausalScaleTTC", "AdamW", "CosineAnnealingLR", "_loss", "clip_grad_norm_"):
        assert _dump(_call(original_run, call)) == _dump(_call(overlap_run, call))
    for method in ("begin_update", "step", "commit_update"):
        original_calls = [
            _dump(node) for node in ast.walk(original_run) if isinstance(node, ast.Call)
        ]
        overlap_calls = [
            _dump(node) for node in ast.walk(overlap_run) if isinstance(node, ast.Call)
        ]
        needle = _dump(_call(original_run, method))
        assert needle in original_calls and needle in overlap_calls


def test_only_admitted_transfer_metric_and_diagnostic_replacements_are_present() -> None:
    source = OVERLAP.read_text(encoding="utf-8")
    assert "DevicePrefetch(data, depth=3)" in source
    assert "prefetch.take(cursor[\"order\"], start, config.batch_size)" in source
    assert "training._record_relational_fg_bg_diagnostic = relational_diagnostic" in source
    assert "losses = scalar_metrics(components, total)" in source
    assert source.count("torch.cuda.current_stream().synchronize()") == 2
    assert "torch.cuda.synchronize()" not in source
    assert '"engineering_freeze_sha256": digest(freeze_path)' in source
    assert '"peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()' in source
    assert 'or cursor["position"] == 0' in source


def test_source_replacement_manifest_and_stage_timing_are_metadata_only() -> None:
    assert set(SOURCE_REPLACEMENTS) == {
        "engineering_admission",
        "input_transfer",
        "relational_diagnostic",
        "scalar_reporting",
        "synchronization",
        "checkpoint_cadence",
        "coordination_pause",
        "runtime_provenance",
        "performance_metadata",
    }
    tree = _tree(OVERLAP)
    run = _definition(tree, "run")
    assert isinstance(run, ast.FunctionDef)
    assignments = {
        target.id
        for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {
        "loss_finished",
        "backward_clip_finished",
        "journal_begin_started",
        "journal_begin_finished",
        "optimizer_submitted",
        "journal_commit_finished",
        "current_stream_sync_finished",
        "stage_ms",
    } <= assignments
    source = OVERLAP.read_text(encoding="utf-8")
    assert source.count("torch.cuda.current_stream().synchronize()") == 2
    assert '"stage_ms": stage_ms' in source


def test_pause_markers_are_user_owned_and_support_both_names(tmp_path: Path) -> None:
    assert pause_request(tmp_path) is None
    stop = tmp_path / "STOP_REQUEST"
    stop.write_text("stop", encoding="utf-8")
    assert pause_request(tmp_path) == stop
    request = tmp_path / "COORDINATION_PAUSE_REQUEST.json"
    request.write_text("{}", encoding="utf-8")
    assert pause_request(tmp_path) == request
    assert stop.exists() and request.exists()


def test_controller_routes_only_admitted_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str, str, list[str]]] = []

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> SimpleNamespace:
        calls.append((output, tag, module, arguments))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(controller_overlap, "_launch", launch)
    controller_overlap.launch(tmp_path, "a5", "operational.train40_system.engine", ["--arm", "a5"])
    controller_overlap.launch(
        tmp_path,
        "h8_features",
        "operational.train40_system.history_features",
        ["--raw-root", "raw", "--kind", "H8"],
    )
    controller_overlap.launch(
        tmp_path, "prepare", "operational.train40_system.prepare_partition", ["--workers", "4"]
    )
    controller_overlap.launch(tmp_path, "teacher", "some.other.module", [])
    assert [call[2] for call in calls] == [
        "operational.train40_system.engine_overlap",
        "operational.train40_system.history_resources8",
        "operational.train40_system.prepare_safe",
        "some.other.module",
    ]
    with pytest.raises(ValueError, match="engine task"):
        controller_overlap.launch(
            tmp_path, "a5", "operational.train40_system.engine", ["--arm", "c2f"]
        )


def test_one_heavy_marker_recognizes_original_and_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        controller_overlap,
        "_active",
        lambda marker: seen.append(marker) is None and marker.endswith("engine_overlap"),
    )
    assert controller_overlap.active("operational.train40_system.engine")
    assert seen == [
        "operational.train40_system.engine",
        "operational.train40_system.engine_overlap",
    ]


def test_freeze_requires_exact_real_admission_and_preserves_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, marker in coordination_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    atomic_json(tmp_path / "TRAINING_PROTOCOL.json", {"fixed": True})
    atomic_json(tmp_path / "ENGINEERING_FREEZE.json", {"fixed": True})
    atomic_json(
        tmp_path / "MODELS_FREEZE.json",
        {
            "files": [],
            "trainer_sha256": digest(ORIGINAL),
            "protocol_sha256": digest(tmp_path / "TRAINING_PROTOCOL.json"),
            "engineering_freeze_sha256": digest(tmp_path / "ENGINEERING_FREEZE.json"),
        },
    )
    admission = tmp_path / "coordination_admission/REAL_COORDINATION_ADMISSION.json"
    atomic_json(admission, {"status": "FAILED", "optimizer_updates": 0})
    monkeypatch.setattr(coordination_freeze, "dependency_files", lambda paths: [])
    with pytest.raises(ValueError, match="must pass"):
        coordination_freeze.run(tmp_path)
    atomic_json(
        admission,
        {
            "status": "PASSED",
            "checkpoint_sha256": "checkpoint",
            "checkpoint_full_state_sha256": "full-checkpoint",
            "exact_loss_and_all_components": True,
            "all_gradient_keys_shapes_and_finite_audited": True,
            "gradient_parity_within_baseline_repeatability": True,
            "exact_loss_all_components_and_all_parameter_gradients": False,
            "repeated_gradient_admission": {"controls_per_variant": 3},
            "prefetch_exact_all_tensor_fields_and_metadata": True,
            "prefetch_preserves_CPU_and_CUDA_RNG": True,
            "optimizer_updates": 0,
            "source_sha256": digest(ROOT / "operational/train40_system/profile_coordination.py"),
        },
    )
    coordination_freeze.run(tmp_path)
    frozen = coordination_freeze.read(tmp_path / "COORDINATION_FREEZE.json")
    assert frozen["models_freeze_sha256"] == digest(tmp_path / "MODELS_FREEZE.json")
    assert frozen["engineering_freeze_sha256"] == digest(tmp_path / "ENGINEERING_FREEZE.json")
    assert frozen["original_engine_sha256"] == digest(ORIGINAL)
    assert frozen["pinned_prefetch_depth"] == 3
