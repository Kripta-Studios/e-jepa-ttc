"""Contracts for fresh all-TRAIN40 RGB producer training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.final_refit.producer_plan import (
    RGB_RUNTIME_SOURCE_MEMBERS,
    build_plan,
    runtime_source_paths,
    write_plan,
)
from operational.final_refit.producers import (
    BoundedRGBPrefetch,
    _final_memory_allowed,
    _validate_producer_contract,
    build_rgb_recipe,
    make_effective_batch,
)
from operational.final_refit.rgb import (
    AllTrainRGBSource,
    _cache_resources_allowed,
    materialize_role_cache,
)
from operational.rgb_port.recipe import ProducerRecipe

ROOT = Path(__file__).resolve().parents[2]


class FakeSource:
    def __init__(self) -> None:
        self.population_size = 6
        self.batch_keys = [("P", 2), ("P", 3), ("H", 2), ("H", 2), ("V", 3), ("V", 3)]
        self.calls: list[tuple[int, ...]] = []

    def batch(self, indices: list[int], modality: str) -> ObjectEventV4Batch:
        assert modality == "rgb"
        self.calls.append(tuple(indices))
        steps = self.batch_keys[indices[0]][1]
        size = len(indices)
        return ObjectEventV4Batch(
            events=torch.zeros(size, steps, 3, 8, 8),
            delta_t_s=torch.full((size, steps - 1), 0.1),
            observable_motion=torch.empty(size, 0),
            visible_heights_px=torch.ones(size, steps),
            target_ttc_s=torch.ones(size),
            boxes_xyxy=torch.ones(size, steps, 4),
            common_square_xyxy=torch.ones(size, 4),
            sequence_ids=["g"] * size,
            sample_tokens=[f"t-{index}" for index in indices],
            track_ids=["track"] * size,
            dinov3_relation_targets=torch.zeros(size, 2, 6, 32, 32),
            dinov3_relation_valid=torch.ones(size, 2, 6, 32, 32, dtype=torch.bool),
        )


def recipe(*, population: int = 6) -> ProducerRecipe:
    return ProducerRecipe(
        fit_id="R_A5",
        base_arm="a5",
        modality="rgb",
        seed=7,
        epochs=18,
        effective_batch_size=32,
        microbatch_size=32,
        accumulation_steps=1,
        updates=18,
        model_config={"in_channels": 3},
        loss_config={},
        training_config={},
        sampler_policy={},
        teacher_policy={},
        authority_sha256="a" * 64,
        role_manifest_sha256="b" * 64,
        producer_population=population,
    )


def test_recipe_is_full_train_fresh_rgb_and_disables_unadmitted_graphs(tmp_path: Path) -> None:
    descriptor = tmp_path / "source.json"
    descriptor.write_text(json.dumps({"schema": "test"}), encoding="utf-8")
    source = SimpleNamespace(
        population_size=88_744,
        teacher=SimpleNamespace(
            manifest_sha256="1" * 64,
            rows_sha256="2" * 64,
        ),
    )
    result = build_rgb_recipe(
        ROOT / "configs/rgb_port/producers.json",
        "R_A5",
        descriptor,
        cast(Any, source),
    )
    assert result.updates == 49_932
    assert result.teacher_policy["roles"] == ["TRAIN"]
    assert result.teacher_policy["native_roles_preserved"] == ["P", "H", "V"]
    assert result.training_config["cuda_graphs"].startswith("DISABLED_UNADMITTED")
    assert result.model_config["modality"] == "rgb"
    assert result.model_config["in_channels"] == 3


def test_effective_batch_preserves_one_boundary_and_splits_role_and_time() -> None:
    source = FakeSource()
    result = make_effective_batch(cast(AllTrainRGBSource, source), list(range(6)), recipe())
    assert source.calls == [(2, 3), (0,), (1,), (4, 5)]
    assert sum(len(part.events) for part in result.parts) == 6
    assert {(len(part.events), int(part.events.shape[1])) for part in result.parts} == {
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
    }


def test_bounded_prefetch_rejects_cursor_or_order_drift() -> None:
    source = FakeSource()
    order = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    cursor: dict[str, Any] = {"order": order, "position": 0}
    loader = BoundedRGBPrefetch(cast(AllTrainRGBSource, source), recipe(), cursor, depth=2)
    try:
        batch = loader.get(order.tolist())
        assert sum(len(part.events) for part in batch.parts) == 6
        cursor["position"] = 1
        cursor["order"] = torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.int64)
        try:
            loader.get([5, 4, 3, 2, 1, 0])
        except RuntimeError as error:
            assert "cursor" in str(error) or "order" in str(error)
        else:  # pragma: no cover - explicit guard assertion
            raise AssertionError("cursor drift was accepted")
    finally:
        loader.close()


def test_effective_batch_rejects_empty_input() -> None:
    try:
        make_effective_batch(cast(AllTrainRGBSource, FakeSource()), [], recipe())
    except ValueError as error:
        assert "empty" in str(error)
    else:  # pragma: no cover - explicit guard assertion
        raise AssertionError("empty effective batch was accepted")


def test_final_cache_materialization_rejects_parallel_workers_before_io(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "missing.json"
    try:
        materialize_role_cache(missing_manifest, tmp_path / "cache", workers=2)
    except ValueError as error:
        assert "exactly one sequential worker" in str(error)
    else:  # pragma: no cover - explicit resource-contract assertion
        raise AssertionError("parallel final cache materialization was accepted")


def test_cache_guard_uses_single_worker_measured_memory_envelope() -> None:
    healthy = {
        "host_available_bytes": 4 * 1024**3,
        "windows_commit_headroom_bytes": 8 * 1024**3,
        "tree_rss_bytes": 4 * 1024**3,
        "disk_free_bytes": 10 * 1024**3,
    }
    assert _cache_resources_allowed(healthy)
    assert not _cache_resources_allowed({**healthy, "host_available_bytes": 4 * 1024**3 - 1})
    assert not _cache_resources_allowed(
        {**healthy, "windows_commit_headroom_bytes": 8 * 1024**3 - 1}
    )
    assert not _cache_resources_allowed({**healthy, "tree_rss_bytes": 4 * 1024**3 + 1})


def test_final_producer_requires_physical_and_commit_headroom() -> None:
    healthy = {
        "host_available_bytes": 7 * 1024**3,
        "windows_commit_headroom_bytes": 9 * 1024**3,
    }
    assert _final_memory_allowed(True, healthy)
    assert not _final_memory_allowed(True, {**healthy, "host_available_bytes": 2 * 1024**3 - 1})
    assert _final_memory_allowed(True, {**healthy, "host_available_bytes": 2 * 1024**3})
    assert not _final_memory_allowed(
        True, {**healthy, "windows_commit_headroom_bytes": 2 * 1024**3 - 1}
    )
    assert _final_memory_allowed(True, {**healthy, "windows_commit_headroom_bytes": 2 * 1024**3})
    assert not _final_memory_allowed(False, healthy)


def test_standalone_producer_plan_binds_inputs_sources_and_stage(tmp_path: Path) -> None:
    cache_bindings = {}
    for role in ("P", "H", "V"):
        binding = tmp_path / f"{role}_CACHE_BINDING.json"
        binding.write_text(json.dumps({"role": role}), encoding="utf-8")
        cache_bindings[role] = binding.name
    descriptor = tmp_path / "source.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "final_refit_rgb_source_descriptor_v1",
                "cache_bindings": cache_bindings,
            }
        ),
        encoding="utf-8",
    )
    teacher = tmp_path / "teacher.json"
    teacher.write_text(
        json.dumps(
            {
                "status": "COMPLETE_VERIFIED",
                "row_count": 88_744,
                "rows_sha256": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    config = ROOT / "configs/rgb_port/producers.json"
    source = SimpleNamespace(
        population_size=88_744,
        teacher=SimpleNamespace(manifest_sha256="1" * 64, rows_sha256="2" * 64),
    )
    producer = build_rgb_recipe(config, "R_A5", descriptor, cast(Any, source))
    raw_config = json.loads(config.read_text(encoding="utf-8"))
    assert (
        raw_config["producers"]["R_A5"]["model_config"].get(
            "deterministic_transport_sampling", False
        )
        is False
    )
    assert producer.model_config["deterministic_transport_sampling"] is True
    assert (
        producer.training_config["deterministic_transport_sampling"]
        == "ENABLED_FINAL_REFIT_PROSPECTIVE"
    )
    executed = {
        "operational/final_refit/producer_plan.py": ROOT
        / "operational/final_refit/producer_plan.py",
        **runtime_source_paths(ROOT),
    }
    runtime_admission = tmp_path / "RUNTIME_ADMISSION.json"

    def sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    admitted_sources = {
        name: sha256(path)
        for name, path in executed.items()
        if name != "operational/final_refit/producer_plan.py"
    }
    admitted_sources["configs/rgb_port/producers.json"] = sha256(config)
    runtime_admission.write_text(
        json.dumps(
            {
                "schema": "final_refit_rgb_eager_runtime_admission_v1",
                "status": "PASS",
                "runtime_admission": True,
                "producer_plan_runtime_admission": "PASS",
                "scientific_updates": 0,
                "source_descriptor_sha256": sha256(descriptor),
                "teacher_manifest_sha256": sha256(teacher),
                "executed_source_sha256": admitted_sources,
            }
        ),
        encoding="utf-8",
    )
    legacy_executed = {
        name: path
        for name, path in executed.items()
        if name
        in {
            "operational/final_refit/producer_plan.py",
            "operational/final_refit/producers.py",
            "operational/final_refit/rgb.py",
            "operational/rgb_port/train_producers.py",
            "operational/rgb_port_revision/loss.py",
        }
    }
    try:
        build_plan(
            source_descriptor=descriptor,
            teacher_manifest=teacher,
            config=config,
            executed_sources=legacy_executed,
            launch_authorized=True,
            runtime_admission=runtime_admission,
        )
    except ValueError as error:
        assert "exact RGB runtime source closure" in str(error)
    else:  # pragma: no cover - explicit legacy-plan rejection
        raise AssertionError("Legacy partial RGB source closure was accepted")
    plan_path = tmp_path / "PRODUCER_PLAN.json"
    plan = build_plan(
        source_descriptor=descriptor,
        teacher_manifest=teacher,
        config=config,
        executed_sources=executed,
        launch_authorized=True,
        runtime_admission=runtime_admission,
    )
    assert set(RGB_RUNTIME_SOURCE_MEMBERS) < set(plan.executed_source_sha256)
    write_plan(
        plan_path,
        plan,
    )
    assert (
        _validate_producer_contract(
            plan_path,
            "R_A5",
            producer,
            source_descriptor=descriptor,
            teacher_manifest=teacher,
            config_path=config,
        )
        == "final_refit_rgb_producer_plan_v1"
    )
    descriptor.write_text(
        json.dumps(
            {
                "schema": "final_refit_rgb_source_descriptor_v1",
                "cache_bindings": cache_bindings,
                "changed": True,
            }
        ),
        encoding="utf-8",
    )
    try:
        _validate_producer_contract(
            plan_path,
            "R_A5",
            producer,
            source_descriptor=descriptor,
            teacher_manifest=teacher,
            config_path=config,
        )
    except ValueError as error:
        assert "input hashes differ" in str(error)
    else:  # pragma: no cover - explicit immutable-input assertion
        raise AssertionError("Changed producer source descriptor was accepted")


def test_standalone_producer_plan_rejects_draft_without_runtime_admission(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "source.json"
    descriptor.write_text(
        json.dumps({"schema": "final_refit_rgb_source_descriptor_v1"}), encoding="utf-8"
    )
    teacher = tmp_path / "teacher.json"
    teacher.write_text(
        json.dumps(
            {
                "status": "COMPLETE_VERIFIED",
                "row_count": 88_744,
                "rows_sha256": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    config = ROOT / "configs/rgb_port/producers.json"
    source = SimpleNamespace(
        population_size=88_744,
        teacher=SimpleNamespace(manifest_sha256="1" * 64, rows_sha256="2" * 64),
    )
    producer = build_rgb_recipe(config, "R_A5", descriptor, cast(Any, source))
    executed = {
        "operational/final_refit/producer_plan.py": ROOT
        / "operational/final_refit/producer_plan.py",
        **runtime_source_paths(ROOT),
    }
    plan_path = tmp_path / "PRODUCER_PLAN_DRAFT.json"
    write_plan(
        plan_path,
        build_plan(
            source_descriptor=descriptor,
            teacher_manifest=teacher,
            config=config,
            executed_sources=executed,
        ),
    )

    try:
        _validate_producer_contract(
            plan_path,
            "R_A5",
            producer,
            source_descriptor=descriptor,
            teacher_manifest=teacher,
            config_path=config,
        )
    except ValueError as error:
        assert "has not passed runtime admission" in str(error)
    else:  # pragma: no cover - explicit authorization-boundary assertion
        raise AssertionError("Draft producer plan was accepted for scientific launch")
