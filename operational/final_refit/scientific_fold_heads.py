"""Scientific PAIR/CTX stages with recursively verified fold-specific ancestors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared

from .fold_training import _stage, _validate_ancestors, _validate_cache, load_fold_training_unit
from .oof_lineage import (
    STAGE_SPEC,
    build_endpoint_lineage,
    load_rotation_identity,
    verify_endpoint_lineage,
    write_endpoint_lineage,
)
from .recipe import FinalRefitRecipe, canonical_sha256, load_recipe, sha256
from .scientific_authority import execution_binding, verify_execution_binding
from .training import CachedTrainingScope, fit_cached_stage


def verify_scientific_admission(path: Path) -> dict[str, Any]:
    """Verify actual endpoint state, exact groups and every scientific ancestor."""
    admission = read_json_shared(path)
    payload = {k: v for k, v in admission.items() if k != "admission_payload_sha256"}
    if (
        admission.get("schema") != "final_refit_fold_endpoint_admission_v1"
        or admission.get("status") != "PASS"
        or admission.get("technical") is not False
        or admission.get("scientific_endpoint") is not True
        or canonical_sha256(payload) != admission.get("admission_payload_sha256")
    ):
        raise ValueError("Invalid scientific endpoint admission")
    lineage_path = path.with_name("STRUCTURAL_LINEAGE.json")
    if sha256(lineage_path) != admission.get("structural_lineage_sha256"):
        raise ValueError("Endpoint structural lineage changed")
    lineage = verify_endpoint_lineage(lineage_path)
    receipt_path = Path(admission["checkpoint_receipt_path"])
    receipt = read_json_shared(receipt_path)
    checkpoint = Path(admission["checkpoint_path"])
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = stored["identity"]
    verify_execution_binding(identity)
    expected_role, expected_dependencies = STAGE_SPEC[lineage["stage_id"]]
    trained_role = "producer_pair" if expected_role == "producer" else "refiner"
    rotation = load_rotation_identity(Path(lineage["fold_contract_path"]), admission["rotation"])
    own = rotation.roles[expected_role]
    generated = rotation.roles["generation"]
    if not (
        receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
        and receipt.get("complete_state") is True
        and receipt.get("completed_updates") == admission["updates"]
        and receipt.get("checkpoint_sha256")
        == sha256(checkpoint)
        == admission["checkpoint_sha256"]
        == lineage["checkpoint_sha256"]
        and sha256(receipt_path) == admission["checkpoint_receipt_sha256"]
        and identity.get("technical") is False
        and identity.get("fit_id") == admission["fit_id"] == lineage["stage_id"]
        and canonical_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})
        == identity.get("identity_sha256")
        == receipt.get("identity_sha256")
        == stored.get("identity_sha256")
        == admission["checkpoint_identity_sha256"]
        and admission["train_group_ids"] == list(own.group_ids)
        and admission["train_rows_sha256"] == own.rows_sha256
        and admission["ordered_sample_token_sha256"] == own.ordered_sample_token_sha256
        and admission["rotation"] == lineage["rotation"]
        and admission["fold_contract_sha256"]
        == rotation.fold_contract_sha256
        == lineage["fold_contract_sha256"]
        and admission["trained_role"] == trained_role == admission["training_cache_role"]
        and admission["rotation_manifest_sha256"] == own.manifest_sha256
        and admission["sample_token_set_sha256"] == own.sample_token_set_sha256
        and admission["ordered_native_ordinal_sha256"] == own.ordered_native_ordinal_sha256
        and admission["generation_group_ids"] == list(generated.group_ids)
        and admission["generation_token_set_sha256"] == generated.sample_token_set_sha256
        and admission["generation_rows_excluded_from_all_ancestors"] is True
        and identity["training_cache_sha256"] == admission["training_cache_sha256"]
        and identity["training_sample_token_sha256"] == own.ordered_sample_token_sha256
    ):
        raise ValueError("Scientific admission differs from actual endpoint or role")
    expected = lineage["training_feature_ancestry"]["direct_parents"]
    if (
        admission["parent_checkpoint_sha256"] != expected
        or set(expected) != set(expected_dependencies)
        or any(
            set(admission[k]) != set(expected)
            for k in ("parent_receipt_sha256", "parent_admission_sha256")
        )
    ):
        raise ValueError("Scientific parent graph differs from transitive lineage")
    refs = admission.get("parent_admission_paths", {})
    if set(refs) != set(expected):
        raise ValueError("Scientific endpoint lacks recursive parent admissions")
    for name, ref in refs.items():
        parent_path = Path(ref)
        if sha256(parent_path) != admission["parent_admission_sha256"][name]:
            raise ValueError("Parent scientific admission changed")
        parent = verify_scientific_admission(parent_path)
        if (
            parent["rotation"] != admission["rotation"]
            or parent["fit_id"] != name
            or parent["checkpoint_sha256"] != expected[name]
            or admission["parent_receipt_sha256"][name] != parent["checkpoint_receipt_sha256"]
        ):
            raise ValueError("Scientific parent belongs to another rotation or stage")
    return admission


def fit_scientific_head(
    *,
    unit_path: Path,
    recipe: FinalRefitRecipe,
    recipe_path: Path,
    stage_id: str,
    cache: Path,
    run: Path,
    parent_receipts: Mapping[str, Path],
    admissions: Mapping[str, Path],
    device: str = "cuda",
    max_updates: int = 100,
) -> dict[str, Any]:
    """Execute a recoverable bounded chunk; COMPLETE requires the fixed full endpoint."""
    if not 1 <= max_updates <= 100:
        raise ValueError("Scientific head chunks must contain 1..100 updates")
    authority = execution_binding(run.parent.parent)
    registered = verify_execution_binding({"execution_authority": authority})
    if sha256(recipe_path) != sha256(Path(registered["recipe"])):
        raise ValueError("Scientific head recipe is not the registered recipe")
    if recipe != load_recipe(recipe_path):
        raise ValueError("In-memory head recipe differs from its immutable file")
    unit = load_fold_training_unit(unit_path)
    stage = _stage(unit, stage_id)
    _validate_cache(unit, stage, cache)
    for path in admissions.values():
        verify_scientific_admission(path)
    checkpoints = _validate_ancestors(unit, stage, parent_receipts, admissions)
    scope = CachedTrainingScope(
        population=stage.population,
        groups=stage.groups,
        role=stage.role,
        experiment_id=f"SCIENTIFIC_OOF_{unit.rotation}",
        split_version=f"TRAIN40_ROTATION_{unit.rotation}_{stage.role}",
        scope_sha256=unit.unit_sha256,
        external_parent_sha256={k: v for k, v in checkpoints.items() if k in {"E_A5", "E_C2F"}},
    )
    receipt = fit_cached_stage(
        recipe=recipe,
        recipe_path=recipe_path,
        stage=stage.recipe(),
        cache_path=cache,
        parent_receipts=parent_receipts,
        run_root=run,
        device_name=device,
        max_updates_this_call=max_updates,
        cache_scope=scope,
        execution_authority=authority,
    )
    if receipt["status"] != "COMPLETE":
        return receipt
    directory = run / "fits" / stage_id
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    lineage = build_endpoint_lineage(
        fold_contract_path=Path(unit.fold_contract_path),
        rotation=unit.rotation,
        stage_id=stage_id,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        checkpoint_receipt_sha256=sha256(receipt_path),
        training_cache_sha256=sha256(cache),
        checkpoint_path=Path(receipt["checkpoint_path"]),
        checkpoint_receipt_path=receipt_path,
        training_cache_path=cache,
        parent_lineage_paths={
            k: p.with_name("STRUCTURAL_LINEAGE.json") for k, p in admissions.items()
        },
    )
    write_endpoint_lineage(directory / "STRUCTURAL_LINEAGE.json", lineage)
    role_name = "producer" if stage.role == "producer_pair" else "refiner"
    own = unit.roles[role_name]
    generated = unit.roles["generation"]
    value: dict[str, Any] = {
        "schema": "final_refit_fold_endpoint_admission_v1",
        "status": "PASS",
        "technical": False,
        "scientific_endpoint": True,
        "rotation": unit.rotation,
        "trained_role": stage.role,
        "fit_id": stage_id,
        "fold_contract_sha256": unit.fold_contract_sha256,
        "rotation_manifest_sha256": unit.rotation_manifest_sha256,
        "train_rows_sha256": own["sha256"],
        "training_cache_sha256": sha256(cache),
        "ordered_sample_token_sha256": own["sample_token_sha256"],
        "sample_token_set_sha256": own["sample_token_set_sha256"],
        "ordered_native_ordinal_sha256": own["ordered_native_ordinal_sha256"],
        "checkpoint_path": receipt["checkpoint_path"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_identity_sha256": receipt["identity_sha256"],
        "checkpoint_receipt_path": str(receipt_path.resolve()),
        "checkpoint_receipt_sha256": sha256(receipt_path),
        "train_group_ids": own["group_ids"],
        "training_cache_role": stage.role,
        "generation_group_ids": generated["group_ids"],
        "generation_token_set_sha256": generated["sample_token_set_sha256"],
        "generation_rows_excluded_from_all_ancestors": True,
        "parent_checkpoint_sha256": checkpoints,
        "parent_receipt_sha256": {k: sha256(p) for k, p in parent_receipts.items()},
        "parent_admission_sha256": {k: sha256(p) for k, p in admissions.items()},
        "parent_admission_paths": {k: str(p.resolve()) for k, p in admissions.items()},
        "updates": stage.updates,
        "structural_lineage_sha256": sha256(directory / "STRUCTURAL_LINEAGE.json"),
    }
    value["admission_payload_sha256"] = canonical_sha256(value)
    atomic_write_json(directory / "ENDPOINT_ADMISSION.json", value)
    verify_scientific_admission(directory / "ENDPOINT_ADMISSION.json")
    return receipt
