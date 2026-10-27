from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from operational.final_refit.features import _canonical_sha256
from operational.final_refit.producer_plan import build_plan, load_plan, write_plan
from operational.final_refit.recipe import (
    FinalRefitRecipe,
    StageRecipe,
    build_recipe,
    load_recipe,
    ordered_row_identity_sha256,
    sha256,
    write_recipe,
)
from operational.final_refit.training import (
    FinalCache,
    PhaseOnlyFinal,
    _balanced_mass,
    _technical_execution_limit,
    _validate_technical_receipt,
    fit_cached_stage,
    learning_rate,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_complete_recipe_has_exposure_scaled_seven_fit_chain(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    recipe = build_recipe(
        source_files={"engine": source},
        external_parent_sha256={"E_A5": "a" * 64, "E_C2F": "b" * 64},
        native_train40_rows_sha256="c" * 64,
        ordered_train40_identity_sha256="d" * 64,
    )
    assert [(item.stage_id, item.updates) for item in recipe.stages] == [
        ("R_A5", 49_932),
        ("R_C2F", 49_932),
        ("PAIR_E_MATCHED", 11_264),
        ("PAIR_R", 11_264),
        ("E_CTX_MATCHED", 12_044),
        ("R_CTX", 12_044),
        ("PHASE_ONLY_EVENT_S7_TRAIN40", 12_044),
    ]
    assert sum(item.updates for item in recipe.stages) == 158_524
    path = tmp_path / "RECIPE.json"
    write_recipe(path, recipe)
    assert load_recipe(path) == recipe


def test_producer_plan_is_only_two_fresh_rgb_endpoints(tmp_path: Path) -> None:
    descriptor = tmp_path / "RGB_SOURCE_DESCRIPTOR.json"
    _write_json(descriptor, {"schema": "final_refit_rgb_source_descriptor_v1"})
    teacher = tmp_path / "TEACHER_MANIFEST.json"
    _write_json(
        teacher,
        {"status": "COMPLETE_VERIFIED", "row_count": 88_744, "rows_sha256": "a" * 64},
    )
    config = tmp_path / "producers.json"
    config.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "producer.py"
    source.write_text("frozen\n", encoding="utf-8")
    plan = build_plan(
        source_descriptor=descriptor,
        teacher_manifest=teacher,
        config=config,
        executed_sources={"producer": source},
    )
    assert [(item.fit_id, item.updates) for item in plan.stages] == [
        ("R_A5", 49_932),
        ("R_C2F", 49_932),
    ]
    assert plan.upper_graph_status == "UNDECIDED_OOF_REVIEW_PENDING"
    assert plan.status == "PREPARED_PENDING_CACHE_BINDINGS_AND_RUNTIME_ADMISSION"
    assert plan.launch_authorized is False
    assert "PAIR_R" in plan.exclusions and "OOF rotations" in plan.exclusions
    path = tmp_path / "PRODUCER_PLAN.json"
    write_plan(path, plan)
    assert load_plan(path) == plan


def test_phase_only_zero_initialization_is_rgb_context_anchor() -> None:
    model = PhaseOnlyFinal().eval()
    features = torch.zeros(3, 340)
    features[:, 160] = torch.tensor([-0.2, 0.1, 0.4])
    widths = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]])
    output = model(features, widths)
    torch.testing.assert_close(output["raw_location"], features[:, 160])
    torch.testing.assert_close(output["correction"], torch.zeros(3, 3))


def test_scaled_upper_learning_rate_has_fixed_endpoints() -> None:
    stage = StageRecipe("R_CTX", "temporal_head", 12_044, 128, 1e-3, (), True, "x")
    assert learning_rate(stage, 1) == 3e-4 / 482
    assert learning_rate(stage, 482) == 3e-4
    assert abs(learning_rate(stage, stage.updates) - 3e-5) < 1e-12


def test_group_mass_is_unit_sum_and_equal_between_unequal_groups() -> None:
    groups = np.asarray(["A", "A", "A", "B"])
    mass = _balanced_mass(groups, expected_groups=2)
    np.testing.assert_allclose(mass, np.asarray([1 / 6, 1 / 6, 1 / 6, 1 / 2], np.float32))
    assert abs(float(mass.sum()) - 1.0) < 1e-6
    assert abs(float(mass[:3].sum()) - float(mass[3])) < 1e-6


def test_technical_training_stays_below_small_scientific_endpoint() -> None:
    assert _technical_execution_limit(
        stage_updates=5, completed_updates=0, requested_updates=5
    ) == 4
    _validate_technical_receipt(
        {
            "status": "IN_PROGRESS",
            "scientific_endpoint": False,
            "completed_updates": 4,
        },
        5,
    )
    with pytest.raises(RuntimeError, match="scientific endpoint"):
        _validate_technical_receipt(
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 5,
            },
            5,
        )


def test_cached_pair_training_resumes_to_fixed_endpoint(tmp_path: Path) -> None:
    stage = StageRecipe(
        "PAIR_R", "pair_head", 5, 4, 1e-4, ("R_A5",), True, "final_refit_pair_cache_v1"
    )
    tokens = np.asarray([f"token-{index}" for index in range(40)])
    groups = np.asarray([f"group-{index}" for index in range(40)])
    recipe = FinalRefitRecipe(
        schema="e_jepa_ttc_final_refit_recipe_v2",
        family="unit",
        seed=7,
        train_queries=40,
        train_groups=40,
        stacking="unit",
        event_branch="unit",
        native_train40_rows_sha256="c" * 64,
        ordered_row_identity_sha256=ordered_row_identity_sha256(
            tokens.tolist(), groups.tolist()
        ),
        stages=(stage,),
        source_sha256={"unit": "a" * 64},
        external_parent_sha256={},
        recipe_sha256="placeholder",
    )
    from operational.final_refit.recipe import canonical_sha256

    recipe = replace(recipe, recipe_sha256=canonical_sha256(recipe.payload_without_hash()))
    recipe_path = tmp_path / "RECIPE.json"
    write_recipe(recipe_path, recipe)
    cache = tmp_path / "pair.npz"
    np.savez(
        cache,
        features=np.zeros((40, 133), dtype=np.float32),
        target_phase=np.linspace(-0.2, 0.2, 40, dtype=np.float32),
        group_id=groups,
        sample_token=tokens,
        sequence_id=groups,
        role=np.full(40, "TRAIN"),
    )
    parent_checkpoint = tmp_path / "parent.pt"
    parent_identity = {"fit_id": "R_A5", "kind": "unit"}
    parent_identity_sha = canonical_sha256(parent_identity)
    torch.save(
        {
            "identity": {**parent_identity, "identity_sha256": parent_identity_sha},
            "identity_sha256": parent_identity_sha,
        },
        parent_checkpoint,
    )
    parent_receipt = tmp_path / "parent.json"
    _write_json(
        parent_receipt,
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "R_A5",
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "identity_sha256": parent_identity_sha,
            "checkpoint_path": str(parent_checkpoint.resolve()),
            "checkpoint_sha256": sha256(parent_checkpoint),
        },
    )
    receipt = {
        "schema": "final_refit_pair_cache_v1",
        "status": "COMPLETE",
        "role": "TRAIN",
        "rows": 40,
        "groups": 40,
        "group_ids": sorted(groups.tolist()),
        "parents": {"R_A5": sha256(parent_checkpoint)},
        "normalizers": {},
        "output": str(cache.resolve()),
        "output_sha256": sha256(cache),
        "compression": "np.savez",
        "targets_used_for_feature_construction": False,
        "native_train40_rows_sha256": recipe.native_train40_rows_sha256,
        "ordered_row_identity_sha256": recipe.ordered_row_identity_sha256,
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    _write_json(cache.with_suffix(".json"), receipt)
    reordered_cache = tmp_path / "pair-reordered.npz"
    np.savez(
        reordered_cache,
        features=np.zeros((40, 133), dtype=np.float32),
        target_phase=np.linspace(-0.2, 0.2, 40, dtype=np.float32),
        group_id=groups[::-1],
        sample_token=tokens[::-1],
        sequence_id=groups[::-1],
        role=np.full(40, "TRAIN"),
    )
    reordered_receipt = {
        **receipt,
        "output": str(reordered_cache.resolve()),
        "output_sha256": sha256(reordered_cache),
    }
    reordered_receipt.pop("receipt_payload_sha256")
    reordered_receipt["receipt_payload_sha256"] = _canonical_sha256(reordered_receipt)
    _write_json(reordered_cache.with_suffix(".json"), reordered_receipt)
    with pytest.raises(ValueError, match="token/group/native order"):
        FinalCache(reordered_cache, stage, recipe)
    other_checkpoint = tmp_path / "other-parent.pt"
    other_identity = {"fit_id": "R_A5", "kind": "other-unit"}
    other_identity_sha = canonical_sha256(other_identity)
    torch.save(
        {
            "identity": {**other_identity, "identity_sha256": other_identity_sha},
            "identity_sha256": other_identity_sha,
        },
        other_checkpoint,
    )
    other_receipt = tmp_path / "other-parent.json"
    _write_json(
        other_receipt,
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "R_A5",
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "identity_sha256": other_identity_sha,
            "checkpoint_path": str(other_checkpoint.resolve()),
            "checkpoint_sha256": sha256(other_checkpoint),
        },
    )
    wrong_fit_receipt = tmp_path / "wrong-fit-parent.json"
    _write_json(
        wrong_fit_receipt,
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "SOME_OTHER_FIT",
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "identity_sha256": other_identity_sha,
            "checkpoint_path": str(other_checkpoint.resolve()),
            "checkpoint_sha256": sha256(other_checkpoint),
        },
    )
    with pytest.raises(ValueError, match="schema, fit identity"):
        fit_cached_stage(
            recipe=recipe,
            recipe_path=recipe_path,
            stage=stage,
            cache_path=cache,
            parent_receipts={"R_A5": wrong_fit_receipt},
            run_root=tmp_path / "wrong-fit-run",
            device_name="cpu",
            max_updates_this_call=1,
        )
    noncanonical_checkpoint = tmp_path / "noncanonical-parent.pt"
    false_identity_sha = "f" * 64
    torch.save(
        {
            "identity": {
                "fit_id": "R_A5",
                "kind": "tampered",
                "identity_sha256": false_identity_sha,
            },
            "identity_sha256": false_identity_sha,
        },
        noncanonical_checkpoint,
    )
    noncanonical_receipt = tmp_path / "noncanonical-parent.json"
    _write_json(
        noncanonical_receipt,
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "R_A5",
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "identity_sha256": false_identity_sha,
            "checkpoint_path": str(noncanonical_checkpoint.resolve()),
            "checkpoint_sha256": sha256(noncanonical_checkpoint),
        },
    )
    with pytest.raises(ValueError, match="checkpoint identity is not canonical"):
        fit_cached_stage(
            recipe=recipe,
            recipe_path=recipe_path,
            stage=stage,
            cache_path=cache,
            parent_receipts={"R_A5": noncanonical_receipt},
            run_root=tmp_path / "noncanonical-run",
            device_name="cpu",
            max_updates_this_call=1,
        )
    technical_checkpoint = tmp_path / "technical-parent.pt"
    technical_identity = {"fit_id": "R_A5", "kind": "unit", "technical": True}
    technical_identity_sha = canonical_sha256(technical_identity)
    torch.save(
        {
            "identity": {
                **technical_identity,
                "identity_sha256": technical_identity_sha,
            },
            "identity_sha256": technical_identity_sha,
        },
        technical_checkpoint,
    )
    technical_parent_receipt = tmp_path / "technical-parent.json"
    _write_json(
        technical_parent_receipt,
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "R_A5",
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "identity_sha256": technical_identity_sha,
            "checkpoint_path": str(technical_checkpoint.resolve()),
            "checkpoint_sha256": sha256(technical_checkpoint),
        },
    )
    with pytest.raises(ValueError, match="technical checkpoint"):
        fit_cached_stage(
            recipe=recipe,
            recipe_path=recipe_path,
            stage=stage,
            cache_path=cache,
            parent_receipts={"R_A5": technical_parent_receipt},
            run_root=tmp_path / "technical-parent-run",
            device_name="cpu",
            max_updates_this_call=1,
        )
    with pytest.raises(ValueError, match="Cache parent checkpoints differ"):
        fit_cached_stage(
            recipe=recipe,
            recipe_path=recipe_path,
            stage=stage,
            cache_path=cache,
            parent_receipts={"R_A5": other_receipt},
            run_root=tmp_path / "wrong-run",
            device_name="cpu",
            max_updates_this_call=1,
        )
    first = fit_cached_stage(
        recipe=recipe,
        recipe_path=recipe_path,
        stage=stage,
        cache_path=cache,
        parent_receipts={"R_A5": parent_receipt},
        run_root=tmp_path / "run",
        device_name="cpu",
        max_updates_this_call=3,
    )
    assert first["completed_updates"] == 3
    second = fit_cached_stage(
        recipe=recipe,
        recipe_path=recipe_path,
        stage=stage,
        cache_path=cache,
        parent_receipts={"R_A5": parent_receipt},
        run_root=tmp_path / "run",
        device_name="cpu",
        max_updates_this_call=2,
    )
    assert second["status"] == "COMPLETE"
    assert second["completed_updates"] == 5
    scratch = fit_cached_stage(
        recipe=recipe,
        recipe_path=recipe_path,
        stage=stage,
        cache_path=cache,
        parent_receipts={"R_A5": parent_receipt},
        run_root=tmp_path / "scratch-run",
        device_name="cpu",
        max_updates_this_call=5,
    )
    assert scratch["status"] == "COMPLETE"
    resumed_payload = torch.load(second["checkpoint_path"], map_location="cpu", weights_only=False)
    scratch_payload = torch.load(scratch["checkpoint_path"], map_location="cpu", weights_only=False)
    assert resumed_payload["completed_updates"] == scratch_payload["completed_updates"] == 5
    assert resumed_payload["cursor"] == scratch_payload["cursor"]
    assert torch.equal(
        resumed_payload["sampler_generator_state"], scratch_payload["sampler_generator_state"]
    )
    assert torch.equal(resumed_payload["torch_rng_state"], scratch_payload["torch_rng_state"])
    for name, value in resumed_payload["model_state_dict"].items():
        assert torch.equal(value, scratch_payload["model_state_dict"][name]), name
    for parameter_id, values in resumed_payload["optimizer_state_dict"]["state"].items():
        other = scratch_payload["optimizer_state_dict"]["state"][parameter_id]
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other[name]), (parameter_id, name)
            else:
                assert value == other[name]
