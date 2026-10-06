"""CPU-only exactness and lifecycle tests for scoped TRAIN40 relation reuse."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from e_jepa_ttc.distillation import dinov3_relational as relational
from operational.efficient_context.common import digest
from operational.train40_system import (
    controller_relation_reuse,
    engine_relation_reuse,
    relation_reuse,
    relation_reuse_freeze,
)
from operational.train40_system.durable_io import atomic_json


def _training_module() -> SimpleNamespace:
    def loss(
        features: torch.Tensor,
        teacher: torch.Tensor,
        valid: torch.Tensor,
        *,
        offsets: tuple[tuple[int, int], ...] = relational.A4_RELATION_OFFSETS,
    ) -> tuple[torch.Tensor, relational.LocalRelationMaps]:
        value = relational.local_relational_distillation_loss(
            features, teacher, valid, offsets=offsets
        )
        with torch.no_grad():
            diagnostic = relational.local_cosine_relation_maps(features, offsets=offsets)
        return value, diagnostic

    return SimpleNamespace(_loss=loss)


def _inputs(*, empty: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(2, 2, 4, 5, 5, generator=generator)
    shape = (2, 2, len(relational.A4_RELATION_OFFSETS), 5, 5)
    teacher = torch.randn(shape, generator=generator)
    valid = torch.zeros(shape, dtype=torch.bool) if empty else torch.ones(shape, dtype=torch.bool)
    return features, teacher, valid


@pytest.mark.parametrize("empty", [False, True])
def test_same_loss_reuse_is_exact_for_finite_and_empty_teacher(empty: bool) -> None:
    features, teacher, valid = _inputs(empty=empty)
    baseline_features = features.clone().requires_grad_(True)
    baseline_module = _training_module()
    baseline_loss, baseline_maps = baseline_module._loss(
        baseline_features, teacher, valid
    )
    baseline_gradient: torch.Tensor | None = None
    if baseline_loss.requires_grad:
        baseline_loss.backward()
        assert baseline_features.grad is not None
        baseline_gradient = baseline_features.grad.detach().clone()

    reused_features = features.clone().requires_grad_(True)
    training = _training_module()
    original_loss = training._loss
    original_maps = relational.local_cosine_relation_maps
    counters = relation_reuse.ReuseCounters()
    with relation_reuse.scoped_relation_reuse(
        cast(relation_reuse.TrainingModule, training), counters
    ):
        reused_loss, reused_maps = training._loss(reused_features, teacher, valid)
        if reused_loss.requires_grad:
            reused_loss.backward()
        assert training._loss is not original_loss
        assert relational.local_cosine_relation_maps is not original_maps
    torch.testing.assert_close(reused_loss, baseline_loss, rtol=0, atol=0)
    torch.testing.assert_close(reused_maps.values, baseline_maps.values, rtol=0, atol=0)
    torch.testing.assert_close(reused_maps.valid, baseline_maps.valid, rtol=0, atol=0)
    if baseline_gradient is None:
        assert reused_features.grad is None
    else:
        torch.testing.assert_close(reused_features.grad, baseline_gradient, rtol=0, atol=0)
    assert reused_maps.values.grad_fn is None
    assert counters.snapshot() == {
        "loss_calls": 1,
        "original_map_calls": 1,
        "reuse_hits": 1,
        "reuse_misses": 0,
        "outside_scope_calls": 0,
    }
    assert training._loss is original_loss
    assert relational.local_cosine_relation_maps is original_maps
    assert relation_reuse.context_is_clear()


def test_offset_mismatch_calls_original_and_does_not_reuse() -> None:
    features, _teacher, _valid = _inputs()
    def loss(values: torch.Tensor) -> None:
        relational.local_cosine_relation_maps(values)
        with torch.no_grad():
            relational.local_cosine_relation_maps(values, offsets=((0, 1),))

    training = SimpleNamespace(_loss=loss)
    counters = relation_reuse.ReuseCounters()
    with relation_reuse.scoped_relation_reuse(
        cast(relation_reuse.TrainingModule, training), counters
    ):
        training._loss(features.requires_grad_(True))
    assert counters.original_map_calls == 2
    assert counters.reuse_hits == 0
    assert counters.reuse_misses == 1


def test_two_loss_calls_have_two_bounded_maps_and_do_not_leak() -> None:
    features, teacher, valid = _inputs()
    training = _training_module()
    counters = relation_reuse.ReuseCounters()
    with relation_reuse.scoped_relation_reuse(
        cast(relation_reuse.TrainingModule, training), counters
    ):
        training._loss(features.clone().requires_grad_(True), teacher, valid)
        assert relation_reuse.context_is_clear()
        training._loss(features.clone().requires_grad_(True), teacher, valid)
        assert relation_reuse.context_is_clear()
    assert counters.loss_calls == 2
    assert counters.original_map_calls == 2
    assert counters.reuse_hits == 2


def test_error_resets_context_and_restores_both_bindings() -> None:
    def loss(features: torch.Tensor) -> None:
        relational.local_cosine_relation_maps(features)
        raise RuntimeError("loss failed")

    training = SimpleNamespace(_loss=loss)
    original_loss = training._loss
    original_maps = relational.local_cosine_relation_maps
    with pytest.raises(RuntimeError, match="loss failed"):
        with relation_reuse.scoped_relation_reuse(
            cast(relation_reuse.TrainingModule, training)
        ):
            training._loss(torch.ones(1, 2, 3, 3, requires_grad=True))
    assert training._loss is original_loss
    assert relational.local_cosine_relation_maps is original_maps
    assert relation_reuse.context_is_clear()
    original_maps(torch.ones(1, 2, 3, 3))


def test_engine_wrapper_restores_bindings_after_delegate_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "RELATION_REUSE_FREEZE.json").write_text("{}", encoding="utf-8")
    (tmp_path / "IDLE_WAIT_FREEZE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine_relation_reuse, "_verify_execution", lambda output: {})
    import e_jepa_ttc.training.causal_scale_eap as training
    from operational.train40_system.checkpoint import DurableState

    original_loss = training._loss
    original_maps = relational.local_cosine_relation_maps
    save_calls: list[str] = []

    def fake_save(state: DurableState, *args: Any, **kwargs: Any) -> None:
        save_calls.append(str(kwargs["status"]))
        state.durable = state.committed

    monkeypatch.setattr(DurableState, "save", fake_save)

    def run(output: Path, arm: str) -> None:
        assert training._loss is not original_loss
        assert relational.local_cosine_relation_maps is not original_maps
        state = DurableState.__new__(DurableState)
        state.directory = output / "fits" / f"{arm}_seed7"
        state.directory.mkdir(parents=True, exist_ok=True)
        state.path = state.directory / "checkpoint_last.pt"
        state.committed = 123
        state.durable = 100
        cast(Any, DurableState.save)(state, status="RUNNING")
        raise RuntimeError("delegate failed")

    monkeypatch.setattr(engine_relation_reuse.engine_idle_wait, "run", run)
    with pytest.raises(RuntimeError, match="delegate failed"):
        engine_relation_reuse.run(tmp_path, "a5")
    assert training._loss is original_loss
    assert relational.local_cosine_relation_maps is original_maps
    assert DurableState.save is fake_save
    assert save_calls == ["RUNNING"]
    assert relation_reuse.context_is_clear()
    runtime = json.loads(
        (tmp_path / "fits/a5_seed7/RELATION_REUSE_RUNTIME.json").read_text("utf-8")
    )
    assert runtime["status"] == "EXITED"
    counters = json.loads(
        (tmp_path / "fits/a5_seed7/RELATION_REUSE_COUNTERS.json").read_text("utf-8")
    )
    assert counters["committed_updates"] == counters["durable_updates"] == 123


def test_controller_routes_only_exact_original_engine_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str, str, list[str]]] = []

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> Any:
        calls.append((output, tag, module, arguments))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(controller_relation_reuse.controller_overlap, "_launch", launch)
    monkeypatch.setattr(controller_relation_reuse, "_launch", launch)
    controller_relation_reuse.launch(
        tmp_path, "a5", "operational.train40_system.engine", ["--arm", "a5"]
    )
    controller_relation_reuse.launch(tmp_path, "teacher", "another.module", [])
    assert [call[2] for call in calls] == [
        "operational.train40_system.engine_relation_reuse",
        "another.module",
    ]
    with pytest.raises(ValueError, match="engine task"):
        controller_relation_reuse.launch(
            tmp_path, "a5", "operational.train40_system.engine", ["--arm", "c2f"]
        )


def test_controller_composes_pause_safe_and_restores_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "COORDINATION_FREEZE.json",
        "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "IDLE_WAIT_FREEZE.json",
        "RELATION_REUSE_FREEZE.json",
    ):
        atomic_json(tmp_path / name, {"files": []})
    original = controller_relation_reuse.controller_overlap.launch
    observed: list[Any] = []

    def run(output: Path, raw: Path, teacher: Path) -> None:
        observed.extend([output, raw, teacher, controller_relation_reuse.controller_overlap.launch])

    monkeypatch.setattr(controller_relation_reuse.controller_pause_safe, "run", run)
    raw, teacher = tmp_path / "raw", tmp_path / "teacher"
    controller_relation_reuse.run(tmp_path, raw, teacher)
    assert observed[:3] == [tmp_path, raw, teacher]
    assert observed[3] is controller_relation_reuse.launch
    assert controller_relation_reuse.controller_overlap.launch is original


def test_freeze_requires_qa_and_binds_real_zero_update_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, marker in relation_reuse_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    coordination = tmp_path / "COORDINATION_FREEZE.json"
    pause = tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    idle = tmp_path / "IDLE_WAIT_FREEZE.json"
    atomic_json(coordination, {"files": [], "trainer_sha256": "trainer"})
    atomic_json(pause, {"files": [], "source_sha256": "pause"})
    atomic_json(idle, {"files": []})
    admission_source = Path(relation_reuse_freeze.__file__).with_name(
        "relation_reuse_admission.py"
    )
    admission = tmp_path / "relation_reuse_admission/REAL_RELATION_REUSE_ADMISSION.json"
    atomic_json(
        admission,
        {
            "status": "PASSED",
            "arm": "a5",
            "checkpoint_sha256": "checkpoint",
            "idle_wait_freeze_sha256": digest(idle),
            "environment": {"OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"},
            "exact_loss_and_all_components": True,
            "all_gradient_keys_shapes_and_finite_audited": True,
            "gradient_tensors_compared": 59,
            "gradient_parity_within_baseline_repeatability": True,
            "gradient_bit_equality_observed": False,
            "repeated_gradient_admission": {"raw": {}, "clipped": {}},
            "first_grad_enabled_map_calls_original": True,
            "same_call_diagnostic_receives_detached_map": True,
            "CPU_and_CUDA_RNG_trajectory_exact": True,
            "optimizer_updates": 0,
            "training_checkpoint_modified": False,
            "source_sha256": digest(admission_source),
            "reuse_counters": {"loss_calls": 3, "reuse_hits": 3},
        },
    )
    monkeypatch.setattr(relation_reuse_freeze, "dependency_files", lambda seeds: [])
    relation_reuse_freeze.run(tmp_path)
    frozen = json.loads((tmp_path / "RELATION_REUSE_FREEZE.json").read_text("utf-8"))
    assert frozen["coordination_freeze_sha256"] == digest(coordination)
    assert frozen["controller_pause_safe_freeze_sha256"] == digest(pause)
    assert frozen["idle_wait_freeze_sha256"] == digest(idle)
    assert frozen["new_optimizer_updates_during_admission"] == 0
    assert frozen["gradient_bit_equality_observed"] is False
    (tmp_path / "RELATION_REUSE_RUFF.txt").write_text("failed", encoding="utf-8")
    (tmp_path / "RELATION_REUSE_FREEZE.json").unlink()
    with pytest.raises(ValueError, match="QA failed"):
        relation_reuse_freeze.run(tmp_path)


def test_admission_source_has_no_optimizer_step_or_profiler() -> None:
    source = Path(relation_reuse_freeze.__file__).with_name(
        "relation_reuse_admission.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert "torch.profiler" not in source
