from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from operational.final_refit.export import (
    export_phase_only_head,
    package_final_predictions,
    seal_final_chain,
)
from operational.final_refit.producer_plan import RGB_RUNTIME_SOURCE_MEMBERS, build_plan, write_plan
from operational.final_refit.recipe import (
    FinalRefitRecipe,
    build_recipe,
    canonical_sha256,
    sha256,
    write_recipe,
)
from operational.final_refit.training import PhaseOnlyFinal


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _feature_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _complete_fixture(
    root: Path,
) -> tuple[
    Path,
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    Path,
]:
    event_receipts: dict[str, Path] = {}
    event_hashes: dict[str, str] = {}
    for name in ("E_A5", "E_C2F"):
        checkpoint = root / "events" / name / "checkpoint_last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(name.encode())
        receipt = checkpoint.with_name("ADMISSION.json")
        _write(
            receipt,
            {
                "schema": "final_refit_external_parent_admission_v1",
                "status": "PASS",
                "reuse_authorized": True,
                "fit_id": name,
                "split_version": "public_TRAIN40_all_train",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "input_manifest_sha256": "a" * 64,
                "teacher_manifest_sha256": "b" * 64,
            },
        )
        event_receipts[name] = receipt
        event_hashes[name] = sha256(checkpoint)
    source = root / "source.py"
    source.write_text("# fixture\n", encoding="utf-8")
    recipe: FinalRefitRecipe = build_recipe(
        source_files={"source": source},
        external_parent_sha256=event_hashes,
        native_train40_rows_sha256="c" * 64,
        ordered_train40_identity_sha256="d" * 64,
    )
    recipe_path = root / "RECIPE.json"
    write_recipe(recipe_path, recipe)
    producer_root = root / "producer"
    producer_root.mkdir()
    cache_bindings: dict[str, str] = {}
    for role in ("P", "H", "V"):
        cache = producer_root / f"{role}.cache"
        cache.write_bytes(role.encode())
        cache_bindings[role] = cache.name
    descriptor = producer_root / "SOURCE.json"
    _write(
        descriptor,
        {
            "schema": "final_refit_rgb_source_descriptor_v1",
            "cache_bindings": cache_bindings,
        },
    )
    teacher = producer_root / "TEACHER.json"
    _write(
        teacher,
        {"status": "COMPLETE_VERIFIED", "row_count": 88_744, "rows_sha256": "1" * 64},
    )
    config = producer_root / "producers.json"
    config.write_text("fixture", encoding="utf-8")
    source_names = ("operational/final_refit/producer_plan.py", *RGB_RUNTIME_SOURCE_MEMBERS)
    producer_sources: dict[str, Path] = {}
    for index, name in enumerate(source_names):
        path = producer_root / f"source_{index}.py"
        path.write_text(name, encoding="utf-8")
        producer_sources[name] = path
    admission = producer_root / "ADMISSION.json"
    admitted_sources = {
        name: sha256(path)
        for name, path in producer_sources.items()
        if name != "operational/final_refit/producer_plan.py"
    }
    admitted_sources["configs/rgb_port/producers.json"] = sha256(config)
    _write(
        admission,
        {
            "schema": "final_refit_rgb_eager_runtime_admission_v1",
            "status": "PASS",
            "runtime_admission": True,
            "producer_plan_runtime_admission": "PASS",
            "scientific_updates": 0,
            "source_descriptor_sha256": sha256(descriptor),
            "teacher_manifest_sha256": sha256(teacher),
            "executed_source_sha256": admitted_sources,
        },
    )
    producer_plan = build_plan(
        source_descriptor=descriptor,
        teacher_manifest=teacher,
        config=config,
        executed_sources=producer_sources,
        launch_authorized=True,
        runtime_admission=admission,
    )
    producer_plan_path = producer_root / "PLAN.json"
    write_plan(producer_plan_path, producer_plan)
    stage_receipts: dict[str, Path] = {}
    cache_receipts: dict[str, Path] = {}
    normalizers: dict[str, Path] = {}
    checkpoint_hashes = dict(event_hashes)
    for declared in recipe.stages:
        fit = root / "fits" / declared.stage_id
        fit.mkdir(parents=True)
        checkpoint = fit / "checkpoint_last.pt"
        if declared.kind == "rgb_producer":
            identity: dict[str, Any] = {
                "schema": "final_refit_rgb_producer_identity_v1",
                "fit_id": declared.stage_id,
                "producer_population": recipe.train_queries,
                "population_role": "TRAIN",
                "historical_roles_preserved": ["P", "H", "V"],
                "initialization": "fresh_seed7_no_event_or_historical_rgb_weights",
                "producer_contract_path": str(producer_plan_path.resolve()),
                "producer_contract_sha256": sha256(producer_plan_path),
                "producer_contract_schema": "final_refit_rgb_producer_plan_v1",
                "source_descriptor_path": str(descriptor.resolve()),
                "source_descriptor_sha256": sha256(descriptor),
                "source_identity": {"fixture": "same-train40-source"},
                "teacher_policy": {
                    "teacher_manifest_sha256": sha256(teacher),
                    "teacher_rows_sha256": "1" * 64,
                },
                "executed_source_sha256": admitted_sources,
            }
        else:
            if declared.stage_id == "E_CTX_MATCHED":
                for modality, context_stage in (
                    ("event", "E_CTX_MATCHED"),
                    ("rgb", "R_CTX"),
                ):
                    path = root / f"{modality}_normalizer.json"
                    context = next(item for item in recipe.stages if item.stage_id == context_stage)
                    payload: dict[str, Any] = {
                        "schema": "final_refit_train40_normalizer_v1",
                        "status": "COMPLETE",
                        "role": "TRAIN",
                        "modality": modality,
                        "rows": 88_744,
                        "parents": {name: checkpoint_hashes[name] for name in context.depends_on},
                        "mean": [0.0] * 17,
                        "scale": [1.0] * 17,
                        "targets_read": False,
                    }
                    payload["normalizer_payload_sha256"] = canonical_sha256(payload)
                    _write(path, payload)
                    normalizers[modality] = path
            cache = fit / "TRAIN_CACHE.npz"
            cache.write_bytes(f"cache:{declared.stage_id}".encode())
            cache_normalizers: dict[str, str] = {}
            if declared.stage_id == "E_CTX_MATCHED":
                cache_normalizers = {"event": sha256(normalizers["event"])}
            elif declared.stage_id == "R_CTX":
                cache_normalizers = {"rgb": sha256(normalizers["rgb"])}
            elif declared.stage_id == "PHASE_ONLY_EVENT_S7_TRAIN40":
                cache_normalizers = {name: sha256(path) for name, path in normalizers.items()}
            cache_receipt = cache.with_suffix(".json")
            cache_payload: dict[str, Any] = {
                "schema": declared.cache_schema,
                "status": "COMPLETE",
                "role": "TRAIN",
                "rows": recipe.train_queries,
                "groups": recipe.train_groups,
                "group_ids": [f"group-{index}" for index in range(recipe.train_groups)],
                "parents": {name: checkpoint_hashes[name] for name in declared.depends_on},
                "normalizers": cache_normalizers,
                "output": str(cache.resolve()),
                "output_sha256": sha256(cache),
                "targets_used_for_feature_construction": False,
            }
            cache_payload["receipt_payload_sha256"] = _feature_sha(cache_payload)
            _write(cache_receipt, cache_payload)
            cache_receipts[declared.stage_id] = cache_receipt
            identity = {
                "schema": "final_refit_cached_stage_identity_v1",
                "fit_id": declared.stage_id,
                "kind": declared.kind,
                "recipe_sha256": recipe.recipe_sha256,
                "technical": False,
                "cache_sha256": sha256(cache),
                "cache_receipt_sha256": sha256(cache_receipt),
                "parent_checkpoint_sha256": {
                    name: checkpoint_hashes[name] for name in declared.depends_on
                },
            }
        identity["identity_sha256"] = canonical_sha256(identity)
        model_state = (
            PhaseOnlyFinal().state_dict()
            if declared.stage_id == "PHASE_ONLY_EVENT_S7_TRAIN40"
            else {}
        )
        torch.save(
            {
                "schema": "rgb_port_producer_checkpoint_v1",
                "identity": identity,
                "identity_sha256": identity["identity_sha256"],
                "model_state_dict": model_state,
            },
            checkpoint,
        )
        receipt = fit / "CHECKPOINT_RECEIPT.json"
        _write(
            receipt,
            {
                "schema": "rgb_port_checkpoint_receipt_v1",
                "fit_id": declared.stage_id,
                "status": "COMPLETE",
                "completed_updates": declared.updates,
                "scientific_endpoint": True,
                "identity_sha256": identity["identity_sha256"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "complete_state": True,
                "accumulation_index": 0,
            },
        )
        stage_receipts[declared.stage_id] = receipt
        checkpoint_hashes[declared.stage_id] = sha256(checkpoint)
    seal_path = root / "MODEL_SEAL.json"
    return recipe_path, stage_receipts, cache_receipts, event_receipts, normalizers, seal_path


def test_complete_chain_seal_and_torchscript_head_parity(tmp_path: Path) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)

    seal = seal_final_chain(
        recipe_path=recipe,
        stage_receipts=stages,
        cache_receipts=caches,
        event_parent_receipts=events,
        normalizers=normalizers,
        output=seal_path,
    )

    assert seal["status"] == "SEALED_NOT_EVALUATED"
    assert seal["components"]["counts"] == {
        "trained_stages": 7,
        "external_event_parents": 2,
        "normalizers": 2,
    }
    assert seal["head_input_contract"]["raw_state"]["shape"] == ["B", 340]
    assert seal["scope"]["end_to_end_sensor_latency_claimed"] is False
    export = tmp_path / "phase_only_final.torchscript.pt"
    receipt = export_phase_only_head(
        recipe_path=recipe,
        fit_directory=stages["PHASE_ONLY_EVENT_S7_TRAIN40"].parent,
        model_seal_path=seal_path,
        output=export,
    )
    assert receipt["status"] == "EXPORTED_PARITY_PASS"
    assert receipt["parity"]["max_abs_difference"] <= 1e-6
    with export.open("rb") as stream:
        model = torch.jit.load(stream)
    assert model(torch.zeros(2, 340), torch.zeros(2, 2)).shape == (2, 4)


def test_package_requires_complete_seal_bound_predictions_and_alias_parity(
    tmp_path: Path,
) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)
    seal_final_chain(
        recipe_path=recipe,
        stage_receipts=stages,
        cache_receipts=caches,
        event_parent_receipts=events,
        normalizers=normalizers,
        output=seal_path,
    )
    tokens = [f"token-{index:04d}" for index in range(6_762)]
    queries = [
        {"sample_token": token, "rgb_history": [f"rgb-{index}"]}
        for index, token in enumerate(tokens)
    ]
    inputs = tmp_path / "INPUTS.json"
    _write(inputs, {"queries": queries, "physical_aliases": {tokens[1]: tokens[0]}})
    predictions = pd.DataFrame(
        {
            "sample_token": tokens,
            "prediction": [1.0, 1.0, *[2.0] * 6_760],
            "rgb_available": [True] * 6_762,
            "fallback_source": ["PHASE_ONLY"] * 6_762,
        }
    )
    predictions_path = tmp_path / "TEST_PREDICTIONS.csv"
    predictions.to_csv(predictions_path, index=False)
    prediction_receipt = tmp_path / "PREDICTIONS_SEALED.json"
    _write(
        prediction_receipt,
        {
            "schema": "final_refit_test12_predictions_v1",
            "status": "COMPLETE",
            "candidate": "PHASE_ONLY_EVENT_S7_TRAIN40",
            "model_seal_sha256": sha256(seal_path),
            "inputs_sha256": sha256(inputs),
            "predictions_sha256": sha256(predictions_path),
            "population": 6_762,
            "computed_under_seal": True,
            "private_targets_read": False,
            "external_submission": False,
            "fallback_policy": "E_CTX_WHEN_RGB_HISTORY_UNAVAILABLE",
            "fallback_count": 0,
        },
    )
    archive = tmp_path / "final_submission.zip"
    package_receipt = tmp_path / "PACKAGE.json"

    result = package_final_predictions(
        inputs_path=inputs,
        predictions_path=predictions_path,
        prediction_receipt_path=prediction_receipt,
        model_seal_path=seal_path,
        output=archive,
        receipt_output=package_receipt,
    )

    assert result["status"] == "PACKAGED_NOT_SUBMITTED"
    assert result["sample_count"] == 6_762
    assert result["aliases_verified"] == 1
    assert result["external_submission"] is False
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["submission.json"]

    broken = json.loads(prediction_receipt.read_text(encoding="utf-8"))
    broken["computed_under_seal"] = False
    broken_receipt = tmp_path / "BROKEN.json"
    _write(broken_receipt, broken)
    with pytest.raises(ValueError, match="seal-bound inference"):
        package_final_predictions(
            inputs_path=inputs,
            predictions_path=predictions_path,
            prediction_receipt_path=broken_receipt,
            model_seal_path=seal_path,
            output=tmp_path / "must_not_exist.zip",
        )
    assert not (tmp_path / "must_not_exist.zip").exists()


def test_seal_rejects_complete_checkpoint_from_another_parent_chain(tmp_path: Path) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)
    receipt_path = stages["E_CTX_MATCHED"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = Path(receipt["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    identity["parent_checkpoint_sha256"]["E_A5"] = "f" * 64
    identity_without_hash = dict(identity)
    identity_without_hash.pop("identity_sha256")
    identity["identity_sha256"] = canonical_sha256(identity_without_hash)
    payload["identity_sha256"] = identity["identity_sha256"]
    torch.save(payload, checkpoint)
    receipt["identity_sha256"] = identity["identity_sha256"]
    receipt["checkpoint_sha256"] = sha256(checkpoint)
    _write(receipt_path, receipt)

    with pytest.raises(ValueError, match="parent checkpoint chain"):
        seal_final_chain(
            recipe_path=recipe,
            stage_receipts=stages,
            cache_receipts=caches,
            event_parent_receipts=events,
            normalizers=normalizers,
            output=seal_path,
        )


def test_seal_rejects_complete_cached_stage_from_another_recipe(tmp_path: Path) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)
    receipt_path = stages["PAIR_R"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = Path(receipt["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    identity["recipe_sha256"] = "f" * 64
    identity_without_hash = dict(identity)
    identity_without_hash.pop("identity_sha256")
    identity["identity_sha256"] = canonical_sha256(identity_without_hash)
    payload["identity_sha256"] = identity["identity_sha256"]
    torch.save(payload, checkpoint)
    receipt["identity_sha256"] = identity["identity_sha256"]
    receipt["checkpoint_sha256"] = sha256(checkpoint)
    _write(receipt_path, receipt)

    with pytest.raises(ValueError, match="training identity differs"):
        seal_final_chain(
            recipe_path=recipe,
            stage_receipts=stages,
            cache_receipts=caches,
            event_parent_receipts=events,
            normalizers=normalizers,
            output=seal_path,
        )


def test_seal_rejects_complete_normalizer_from_another_chain(tmp_path: Path) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)
    normalizer = json.loads(normalizers["rgb"].read_text(encoding="utf-8"))
    normalizer["parents"]["R_A5"] = "f" * 64
    normalizer.pop("normalizer_payload_sha256")
    normalizer["normalizer_payload_sha256"] = canonical_sha256(normalizer)
    _write(normalizers["rgb"], normalizer)

    with pytest.raises(ValueError, match="rgb normalizer parent chain"):
        seal_final_chain(
            recipe_path=recipe,
            stage_receipts=stages,
            cache_receipts=caches,
            event_parent_receipts=events,
            normalizers=normalizers,
            output=seal_path,
        )


def test_seal_rejects_alternate_normalizer_with_same_parents(tmp_path: Path) -> None:
    recipe, stages, caches, events, normalizers, seal_path = _complete_fixture(tmp_path)
    normalizer = json.loads(normalizers["rgb"].read_text(encoding="utf-8"))
    normalizer["mean"][0] = 0.25
    normalizer.pop("normalizer_payload_sha256")
    normalizer["normalizer_payload_sha256"] = canonical_sha256(normalizer)
    _write(normalizers["rgb"], normalizer)

    with pytest.raises(ValueError, match="cache normalizers differ"):
        seal_final_chain(
            recipe_path=recipe,
            stage_receipts=stages,
            cache_receipts=caches,
            event_parent_receipts=events,
            normalizers=normalizers,
            output=seal_path,
        )
