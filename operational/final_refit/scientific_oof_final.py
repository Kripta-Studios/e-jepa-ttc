"""Finish the scientific TRAIN40 model from physical five-fold OOF features.

This module deliberately separates two graphs:

* ``training_feature_ancestry`` contains only physical OOF fragments generated
  by fold endpoints whose complete scientific admissions are recursively
  verified;
* ``deployment_dependencies`` contains the full TRAIN40 producers and freshly
  fitted deployment PAIR endpoints used after training.

The two graphs are recorded together but are never represented as checkpoint
parents of one another.  In particular, the context and PHASE_ONLY fits use a
``StageRecipe`` with no checkpoint parents and a cache receipt whose parents
are OOF authority hashes.  This avoids fabricating conventional ancestry for
stacked OOF training.
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import json
import os
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader
from operational.rgb_port.accounting import atomic_write_json

from .event_parity import load_v12_event_producer
from .materialize import materialize_query_pair_features
from .oof_lineage import (
    build_deployment_transition,
    build_stitch_receipt,
    verify_stitch_receipt,
)
from .oof_stitch import (
    PHASE17_RAW_EVENT,
    PHASE17_RAW_RGB,
    PHASE20_CONTEXT,
    fit_oof_phase17_normalizer,
    payload_sha256,
    stitch_from_verified_lineage,
    write_oof_stitch,
)
from .producers import build_rgb_recipe, load_rgb_endpoint
from .recipe import FinalRefitRecipe, StageRecipe, canonical_sha256, load_recipe, sha256, stage
from .rgb import load_all_train_source
from .scientific_authority import execution_binding, load_config, verify_execution_binding
from .scientific_fold_heads import verify_scientific_admission
from .scientific_oof_features import (
    normalize_histories,
    physical_phase20_to_state340,
    seal_role_cache,
)
from .test12 import (
    OFFICIAL_TEST_QUERIES,
    TRUSTED_EVENT_INPUT_FREEZE_SHA256,
    TRUSTED_INPUTS_SHA256,
    TRUSTED_MODEL_INPUT_FREEZE_SHA256,
    TRUSTED_OFFICIAL_TEST_INPUTS_SHA256,
    FinalV16Train40Predictor,
    PredictionFragments,
    Test12Predictor,
    validate_test12_inputs,
)
from .training import CachedTrainingScope, fit_cached_stage, load_cached_endpoint

SCHEMA = "scientific_oof_stage_receipt_v1"
SEAL_SCHEMA = "scientific_oof_final_model_seal_v1"
EXPORT_SCHEMA = "scientific_oof_phase_only_torchscript_v1"
PREDICTION_SCHEMA = "scientific_oof_test12_prediction_receipt_v1"
FINAL_STAGES = ("PAIR_E_MATCHED", "PAIR_R", "E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40")
INTERFACE_LAYOUT = {
    PHASE17_RAW_EVENT: ("E_CTX", ("E_A5", "E_C2F", "PAIR_E")),
    PHASE17_RAW_RGB: ("R_CTX", ("R_A5", "R_C2F", "PAIR_R")),
    PHASE20_CONTEXT: ("PHASE20", ("E_CTX", "R_CTX")),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace changed scientific artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _config(run: Path) -> dict[str, Path | str]:
    value = load_config(run.resolve())
    required = {
        "fold_contract",
        "config",
        "rgb_source",
        "teacher",
        "event_root",
        "raw_root",
        "recipe",
        "workspace",
        "active_rgb_runtime",
        "device",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"SCIENTIFIC_CONFIG lacks keys: {sorted(missing)}")
    result: dict[str, Path | str] = {"device": str(value["device"])}
    if result["device"] not in {"cpu", "cuda"}:
        raise ValueError("SCIENTIFIC_CONFIG device must be cpu or cuda")
    for name in required - {"device"}:
        result[name] = Path(str(value[name])).resolve(strict=True)
    for name in (
        "test12_inputs",
        "official_test_inputs",
        "model_input_freeze",
        "event_input_freeze",
        "v12_run",
        "v12_protocol",
    ):
        if name in value:
            result[name] = Path(str(value[name])).resolve(strict=True)
    return result


def _native_supervision(rgb_source: Path) -> dict[str, np.ndarray]:
    descriptor = _read(rgb_source)
    base = rgb_source.parent
    frames = []
    for role in ("P", "H", "V"):
        raw = Path(str(descriptor["role_manifests"][role]))
        manifest_path = (raw if raw.is_absolute() else base / raw).resolve(strict=True)
        manifest = _read(manifest_path)
        rows_path = Path(str(manifest["rows_path"])).resolve(strict=True)
        if sha256(rows_path) != manifest["rows_sha256"]:
            raise ValueError("TRAIN40 supervision manifest rows changed")
        frames.append(
            pd.read_parquet(
                rows_path,
                columns=[
                    "source_ordinal",
                    "sample_token",
                    "group_id",
                    "sequence_id",
                    "target_phase",
                ],
            )
        )
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("source_ordinal", kind="stable")
        .reset_index(drop=True)
    )
    if (
        len(frame) != 88_744
        or frame.source_ordinal.astype(int).tolist() != list(range(88_744))
        or frame.sample_token.astype(str).duplicated().any()
        or frame.group_id.astype(str).nunique() != 40
    ):
        raise ValueError("Supervision rows are not the exact unique TRAIN40 native order")
    target = frame.target_phase.to_numpy(np.float32)
    if not np.isfinite(target).all():
        raise ValueError("TRAIN40 supervision contains nonfinite phase")
    return {
        "sample_token": frame.sample_token.astype(str).to_numpy(),
        "group_id": frame.group_id.astype(str).to_numpy(),
        "sequence_id": frame.sequence_id.astype(str).to_numpy(),
        "target_phase": target,
        "role": np.full(len(frame), "TRAIN"),
    }


def _fragment_paths(
    run: Path, interface: str
) -> tuple[dict[int, Path], dict[int, dict[str, Path]]]:
    directory, endpoints = INTERFACE_LAYOUT[interface]
    fragments: dict[int, Path] = {}
    lineages: dict[int, dict[str, Path]] = {}
    for rotation in range(5):
        root = run / "rotations" / str(rotation)
        fragment = root / "generation" / directory / "fragment.npz"
        receipt_path = fragment.with_suffix(".json")
        receipt = _read(receipt_path.resolve(strict=True))
        if (
            receipt.get("output_sha256") != sha256(fragment.resolve(strict=True))
            or receipt.get("targets_used") is not False
        ):
            raise ValueError(f"Rotation {rotation} {interface} fragment is not complete")
        references: dict[str, Path] = {}
        for endpoint in endpoints:
            admission_path = root / "fits" / endpoint / "ENDPOINT_ADMISSION.json"
            admission = verify_scientific_admission(admission_path.resolve(strict=True))
            if admission["rotation"] != rotation or admission["fit_id"] != endpoint:
                raise ValueError("Generation endpoint admission belongs to another rotation")
            lineage = admission_path.with_name("STRUCTURAL_LINEAGE.json")
            if sha256(lineage) != admission["structural_lineage_sha256"]:
                raise ValueError("Generation endpoint lineage changed after scientific admission")
            references[endpoint] = lineage
        declared = receipt.get("parents")
        expected_checkpoints = {
            endpoint: verify_scientific_admission(
                root / "fits" / endpoint / "ENDPOINT_ADMISSION.json"
            )["checkpoint_sha256"]
            for endpoint in endpoints
        }
        if declared != expected_checkpoints:
            raise ValueError("Generation fragment parents differ from scientific endpoints")
        fragments[rotation] = fragment
        lineages[rotation] = references
    return fragments, lineages


def _authority(run: Path, fold_contract: Path, interface: str) -> Path:
    root = run / "final" / "oof"
    output = root / f"{interface}.authority.json"
    fragments, lineages = _fragment_paths(run, interface)
    value = build_stitch_receipt(
        fold_contract_path=fold_contract,
        interface=interface,
        fragment_payloads=fragments,
        endpoint_lineages=lineages,
    )
    _write_once(output, value)
    verify_stitch_receipt(output)
    return output


def _stitch(
    run: Path, fold_contract: Path, interface: str, supervision: Mapping[str, np.ndarray]
) -> Path:
    authority = _authority(run, fold_contract, interface)
    values, receipt = stitch_from_verified_lineage(
        authority,
        np.asarray(supervision["sample_token"]).astype(str).tolist(),
        np.asarray(supervision["group_id"]).astype(str).tolist(),
    )
    output = run / "final" / "oof" / f"{interface}.npz"
    if not output.exists():
        write_oof_stitch(output, values, receipt)
    else:
        stored = _read(output.with_suffix(".json"))
        with np.load(output, allow_pickle=False) as archive:
            existing = {name: archive[name] for name in archive.files}
        if (
            stored.get("lineage_authority_sha256")
            != verify_stitch_receipt(authority)["stitch_sha256"]
            or stored.get("output_sha256") != sha256(output)
            or stored.get("payload_sha256") != payload_sha256(existing)
            or set(existing) != set(values)
            or any(not np.array_equal(existing[name], values[name]) for name in values)
        ):
            raise ValueError("Existing OOF stitch belongs to another authority")
    return output


def prepare_oof_training(run: Path) -> dict[str, Any]:
    """Build verified stitches, unique PHASE17 normalizers, and final training caches."""
    run = run.resolve()
    config = _config(run)
    fold_contract = cast(Path, config["fold_contract"])
    recipe = load_recipe(cast(Path, config["recipe"]))
    supervision = _native_supervision(cast(Path, config["rgb_source"]))
    paths = {
        interface: _stitch(run, fold_contract, interface, supervision)
        for interface in (PHASE17_RAW_EVENT, PHASE17_RAW_RGB, PHASE20_CONTEXT)
    }
    normalizers: dict[str, Path] = {}
    for modality, interface in (("event", PHASE17_RAW_EVENT), ("rgb", PHASE17_RAW_RGB)):
        output = run / "final" / "oof" / f"{modality}_normalizer.json"
        temporary = output.with_suffix(f".{os.getpid()}.expected.json")
        expected = fit_oof_phase17_normalizer(paths[interface], temporary)
        try:
            if output.exists():
                if _read(output) != expected:
                    raise ValueError(f"Existing {modality} OOF normalizer changed")
            else:
                os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        normalizers[modality] = output

    caches: dict[str, Path] = {}
    for stage_id, modality, interface in (
        ("E_CTX_MATCHED", "event", PHASE17_RAW_EVENT),
        ("R_CTX", "rgb", PHASE17_RAW_RGB),
    ):
        with np.load(paths[interface], allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        normalizer = _read(normalizers[modality])
        values = {
            "features": normalize_histories(
                raw,
                np.asarray(normalizer["mean"], np.float32),
                np.asarray(normalizer["scale"], np.float32),
            ),
            "timing": np.asarray(raw["timing"], np.float32),
            "valid": np.asarray(raw["valid"], np.bool_),
            "expert_phase": np.asarray(raw["expert_phase"], np.float32),
            "target_phase": supervision["target_phase"],
            "group_id": supervision["group_id"],
            "sample_token": supervision["sample_token"],
            "role": supervision["role"],
        }
        output = run / "final" / "cache" / f"{stage_id}.npz"
        seal_role_cache(
            output,
            values,
            schema=cast(str, stage(recipe, stage_id).cache_schema),
            role="TRAIN",
            parents={"OOF_AUTHORITY": sha256(paths[interface].with_suffix(".json"))},
            rows_sha=recipe.native_train40_rows_sha256,
            normalizers={modality: sha256(normalizers[modality])},
            details={
                "training_feature_domain": "FIVE_FOLD_PHYSICAL_OOF",
                "deployment_parent_checkpoints": [],
            },
        )
        caches[stage_id] = output

    with np.load(paths[PHASE20_CONTEXT], allow_pickle=False) as archive:
        phase = {name: archive[name] for name in archive.files}
    phase_output = run / "final" / "cache" / "PHASE_ONLY_EVENT_S7_TRAIN40.npz"
    seal_role_cache(
        phase_output,
        {
            "features": physical_phase20_to_state340(np.asarray(phase["features"], np.float32)),
            "rgb_width_logits": np.asarray(phase["rgb_width_logits"], np.float32),
            "target_phase": supervision["target_phase"],
            "group_id": supervision["group_id"],
            "sample_token": supervision["sample_token"],
            "role": supervision["role"],
        },
        schema=cast(str, stage(recipe, "PHASE_ONLY_EVENT_S7_TRAIN40").cache_schema),
        role="TRAIN",
        parents={"OOF_AUTHORITY": sha256(paths[PHASE20_CONTEXT].with_suffix(".json"))},
        rows_sha=recipe.native_train40_rows_sha256,
        normalizers={modality: sha256(path) for modality, path in normalizers.items()},
        details={
            "training_feature_domain": "FIVE_FOLD_PHYSICAL_OOF",
            "physical_phase20_adapter": "ZEROS_TO_340_SELECTED_COORDINATES_ONLY",
            "deployment_parent_checkpoints": [],
        },
    )
    caches["PHASE_ONLY_EVENT_S7_TRAIN40"] = phase_output
    receipt = {
        "schema": "scientific_oof_final_preparation_v1",
        "status": "COMPLETE",
        "fold_contract_sha256": sha256(fold_contract),
        "stitches": {name: sha256(path) for name, path in paths.items()},
        "normalizers": {name: sha256(path) for name, path in normalizers.items()},
        "caches": {name: sha256(path) for name, path in caches.items()},
        "targets_used_for_feature_construction": False,
        "targets_attached_only_after_oof_stitch": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_once(run / "final" / "OOF_PREPARATION.json", receipt)
    return receipt


def _oof_stage(recipe: FinalRefitRecipe, stage_id: str) -> StageRecipe:
    if stage_id not in {"E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40"}:
        raise ValueError("Only final context/PHASE stages consume OOF caches")
    return replace(stage(recipe, stage_id), depends_on=())


def prepare_deployment_rgb_pair(run: Path) -> dict[str, Any]:
    """Materialize the full TRAIN40 RGB A5 PAIR cache with durable fragments."""
    run = run.resolve(strict=True)
    config = _config(run)
    recipe = load_recipe(cast(Path, config["recipe"]))
    output = run / "final/cache/PAIR_R.npz"
    receipt_path = cast(Path, config["active_rgb_runtime"]) / "fits/R_A5/CHECKPOINT_RECEIPT.json"
    parent = _read(receipt_path.resolve(strict=True))
    checkpoint = Path(str(parent["checkpoint_path"])).resolve(strict=True)
    if (
        parent.get("status") != "COMPLETE"
        or parent.get("scientific_endpoint") is not True
        or parent.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise ValueError("Full TRAIN40 R_A5 endpoint is incomplete or changed")
    source = load_all_train_source(cast(Path, config["rgb_source"]), cast(Path, config["teacher"]))
    try:
        producer_recipe = build_rgb_recipe(
            cast(Path, config["config"]),
            "R_A5",
            cast(Path, config["rgb_source"]),
            source,
        )
        model = load_rgb_endpoint(
            cast(Path, config["active_rgb_runtime"]) / "fits/R_A5",
            producer_recipe,
            device=cast(str, config["device"]),
        )
        from .scientific_oof_pipeline import make_guard

        return materialize_query_pair_features(
            source,
            model,
            output,
            modality="rgb",
            parents={"R_A5": str(parent["checkpoint_sha256"])},
            native_train40_rows_sha256=recipe.native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=recipe.ordered_row_identity_sha256,
            device=cast(str, config["device"]),
            guard=make_guard(run),
        )
    finally:
        source.close()


def fit_next_oof_stage(run: Path, *, max_updates: int = 100) -> dict[str, Any]:
    """Advance the first unfinished final OOF-trained stage by a bounded chunk."""
    run = run.resolve()
    config = _config(run)
    recipe_path = cast(Path, config["recipe"])
    recipe = load_recipe(recipe_path)
    preparation_path = run / "final" / "OOF_PREPARATION.json"
    preparation = _read(preparation_path.resolve(strict=True))
    if preparation.get("status") != "COMPLETE":
        raise ValueError("OOF preparation is incomplete")
    from .scientific_oof_pipeline import make_guard

    guard = make_guard(run)
    stages = (
        "PAIR_E_MATCHED",
        "PAIR_R",
        "E_CTX_MATCHED",
        "R_CTX",
        "PHASE_ONLY_EVENT_S7_TRAIN40",
    )
    for stage_id in stages:
        receipt_path = run / "final" / "fits" / stage_id / "CHECKPOINT_RECEIPT.json"
        if receipt_path.exists() and _read(receipt_path).get("status") == "COMPLETE":
            continue
        cache = (
            run / "cache" / "PAIR_E_MATCHED.npz"
            if stage_id == "PAIR_E_MATCHED"
            else run / "final" / "cache" / f"{stage_id}.npz"
        )
        cache_receipt = _read(cache.with_suffix(".json"))
        is_oof = stage_id not in {"PAIR_E_MATCHED", "PAIR_R"}
        scope = (
            CachedTrainingScope(
                population=recipe.train_queries,
                groups=recipe.train_groups,
                role="TRAIN",
                experiment_id="SCIENTIFIC_OOF_FINAL_TRAIN40",
                split_version="TRAIN40_FIVE_FOLD_PHYSICAL_OOF",
                scope_sha256=canonical_sha256(
                    {
                        "preparation": sha256(preparation_path),
                        "cache": sha256(cache),
                        "parents": cache_receipt["parents"],
                    }
                ),
                external_parent_sha256={},
            )
            if is_oof
            else None
        )
        parent_receipts: dict[str, Path] = {}
        if stage_id == "PAIR_E_MATCHED":
            parent_receipts = {"E_A5": run / "parents" / "E_A5_ADMISSION.json"}
        elif stage_id == "PAIR_R":
            parent_receipts = {
                "R_A5": cast(Path, config["active_rgb_runtime"])
                / "fits"
                / "R_A5"
                / "CHECKPOINT_RECEIPT.json"
            }
        guard()
        receipt = fit_cached_stage(
            recipe=recipe,
            recipe_path=recipe_path,
            stage=_oof_stage(recipe, stage_id) if is_oof else stage(recipe, stage_id),
            cache_path=cache,
            parent_receipts=parent_receipts,
            run_root=run / "final",
            device_name=cast(str, config["device"]),
            max_updates_this_call=max_updates,
            cache_scope=scope,
            execution_authority=execution_binding(run),
        )
        if receipt.get("status") != "COMPLETE":
            return _write_final_receipt(run, stage_id, receipt, max_updates)
    return _write_final_receipt(run, "final", {"status": "COMPLETE", "completed_updates": 0}, 0)


def _write_final_receipt(
    run: Path, stage_id: str, receipt: Mapping[str, Any], updates_this_call: int
) -> dict[str, Any]:
    del stage_id, receipt
    required = (
        "PAIR_E_MATCHED",
        "PAIR_R",
        "E_CTX_MATCHED",
        "R_CTX",
        "PHASE_ONLY_EVENT_S7_TRAIN40",
    )
    fit_receipts = [run / "final/fits" / name / "CHECKPOINT_RECEIPT.json" for name in required]
    completed = sum(
        int(_read(path).get("completed_updates", 0)) for path in fit_receipts if path.exists()
    )
    heads_complete = all(
        (run / "final" / "fits" / name / "CHECKPOINT_RECEIPT.json").exists()
        and _read(run / "final" / "fits" / name / "CHECKPOINT_RECEIPT.json").get("status")
        == "COMPLETE"
        for name in required
    )
    admissions = {
        str(path.resolve()): sha256(path)
        for path in sorted(run.glob("rotations/*/fits/*/ENDPOINT_ADMISSION.json"))
    }
    fragments = {
        str(path.resolve()): sha256(path)
        for path in sorted(run.glob("rotations/*/generation/*/fragment.npz"))
    }
    expected_admissions = {
        str((run / f"rotations/{rotation}/fits/{name}/ENDPOINT_ADMISSION.json").resolve())
        for rotation in range(5)
        for name in ("E_A5", "E_C2F", "R_A5", "R_C2F", "PAIR_E", "PAIR_R", "E_CTX", "R_CTX")
    }
    expected_fragments = {
        str((run / f"rotations/{rotation}/generation/{name}/fragment.npz").resolve())
        for rotation in range(5)
        for name in ("E_CTX", "R_CTX", "PHASE20")
    }
    inputs_complete = (
        set(admissions) == expected_admissions and set(fragments) == expected_fragments
    )
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "stage_id": "FINAL",
        "status": "RUNNING",
        "scientific_endpoint": False,
        "all_head_endpoints_complete": heads_complete,
        "all_oof_inputs_complete": inputs_complete,
        "admissions_sha256": admissions,
        "fragment_files_sha256": fragments,
        "fragment_sha256": canonical_sha256(fragments),
        "checkpoint_updates": 100,
        "completed_updates": completed,
        "durable_updates": completed,
        "pending_update_upper": 0,
        "updates_this_call": int(updates_this_call),
        "manual_stop": (run / "STOP_REQUEST").exists(),
    }
    value["receipt_sha256"] = canonical_sha256(value)
    atomic_write_json(run / "final" / "FINAL_RECEIPT.json", value)
    return value


def _complete_final_receipt(run: Path, artifacts: Mapping[str, Path]) -> dict[str, Any]:
    required = {
        "model_seal",
        "head_export",
        "head_export_receipt",
        "predictions",
        "prediction_receipt",
        "submission_zip",
        "package_receipt",
    }
    if set(artifacts) != required:
        raise ValueError("Final completion requires the exact seven deliverables")
    value = _write_final_receipt(run, "FINAL", {}, 0)
    if not value["all_head_endpoints_complete"] or not value["all_oof_inputs_complete"]:
        raise ValueError("Final completion requires all endpoints and OOF inputs")
    resolved = {name: path.resolve(strict=True) for name, path in artifacts.items()}
    seal, seal_sha = _scientific_seal(resolved["model_seal"])
    export = _read(resolved["head_export_receipt"])
    prediction = _read(resolved["prediction_receipt"])
    package = _read(resolved["package_receipt"])
    verified_package = package_scientific_predictions(run)
    if (
        seal.get("status") != "SEALED_NOT_EVALUATED"
        or export.get("status") != "EXPORTED_NOT_EVALUATED"
        or export.get("model_seal_sha256") != seal_sha
        or export.get("output_sha256") != sha256(resolved["head_export"])
        or prediction.get("status") != "COMPLETE"
        or prediction.get("model_seal_sha256") != seal_sha
        or prediction.get("predictions_sha256") != sha256(resolved["predictions"])
        or prediction.get("private_targets_read") is not False
        or package.get("status") != "PACKAGED_NOT_SUBMITTED"
        or package.get("model_seal_sha256") != seal_sha
        or package.get("prediction_receipt_sha256") != sha256(resolved["prediction_receipt"])
        or package.get("zip_sha256") != sha256(resolved["submission_zip"])
        or package.get("external_submission") is not False
        or verified_package != package
    ):
        raise ValueError("Final deliverables are incomplete or cross-bound incorrectly")
    value.update(
        {
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "artifact_paths": {name: str(path) for name, path in resolved.items()},
            "artifact_sha256": {name: sha256(path) for name, path in resolved.items()},
            "manual_stop": False,
        }
    )
    value["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    atomic_write_json(run / "final/FINAL_RECEIPT.json", value)
    return value


def seal_scientific_final(
    run: Path,
    *,
    deployment_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    """Seal OOF training ancestry and distinct full-fit deployment dependencies."""
    run = run.resolve()
    config = _config(run)
    recipe_path = cast(Path, config["recipe"])
    recipe = load_recipe(recipe_path)
    expected_dependencies = {
        "E_A5",
        "E_C2F",
        "PAIR_E",
        "R_A5",
        "R_C2F",
        "PAIR_R",
        "E_CTX_MATCHED",
        "R_CTX",
    }
    if set(deployment_dependencies) != expected_dependencies or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in deployment_dependencies.values()
    ):
        raise ValueError("Scientific seal requires the exact full-fit deployment graph")
    fits: dict[str, dict[str, Any]] = {}
    for stage_id in ("E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40"):
        declared = _oof_stage(recipe, stage_id)
        directory = run / "final" / "fits" / stage_id
        receipt_path = directory / "CHECKPOINT_RECEIPT.json"
        receipt = _read(receipt_path.resolve(strict=True))
        checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = stored.get("identity", {})
        verify_execution_binding(identity)
        identity_payload = dict(identity)
        claimed_identity = identity_payload.pop("identity_sha256", None)
        cache = run / "final/cache" / f"{stage_id}.npz"
        cache_receipt_path = cache.with_suffix(".json")
        cache_receipt = _read(cache_receipt_path.resolve(strict=True))
        current_normalizers = {
            name: sha256(run / f"final/oof/{name}_normalizer.json") for name in ("event", "rgb")
        }
        expected_normalizers = (
            {"event": current_normalizers["event"]}
            if stage_id == "E_CTX_MATCHED"
            else (
                {"rgb": current_normalizers["rgb"]} if stage_id == "R_CTX" else current_normalizers
            )
        )
        interface = {
            "E_CTX_MATCHED": PHASE17_RAW_EVENT,
            "R_CTX": PHASE17_RAW_RGB,
            "PHASE_ONLY_EVENT_S7_TRAIN40": PHASE20_CONTEXT,
        }[stage_id]
        expected_cache_parents = {"OOF_AUTHORITY": sha256(run / f"final/oof/{interface}.json")}
        if (
            receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("scientific_endpoint") is not True
            or receipt.get("complete_state") is not True
            or receipt.get("fit_id") != stage_id
            or int(receipt.get("completed_updates", -1)) != declared.updates
            or receipt.get("checkpoint_sha256") != sha256(checkpoint)
            or receipt.get("identity_sha256") != claimed_identity
            or stored.get("identity_sha256") != claimed_identity
            or identity.get("identity_sha256") != claimed_identity
            or canonical_sha256(identity_payload) != claimed_identity
            or identity.get("schema") != "final_refit_cached_stage_identity_v1"
            or identity.get("fit_id") != stage_id
            or identity.get("kind") != declared.kind
            or identity.get("recipe_sha256") != recipe.recipe_sha256
            or identity.get("training_cache_sha256") != sha256(cache)
            or identity.get("cache_receipt_sha256") != sha256(cache_receipt_path)
            or cache_receipt.get("schema") != declared.cache_schema
            or cache_receipt.get("status") != "COMPLETE"
            or cache_receipt.get("role") != "TRAIN"
            or cache_receipt.get("output_sha256") != sha256(cache)
            or cache_receipt.get("normalizers") != expected_normalizers
            or cache_receipt.get("parents") != expected_cache_parents
            or identity.get("parent_checkpoint_sha256") != {}
            or identity.get("cache_scope", {}).get("split_version")
            != "TRAIN40_FIVE_FOLD_PHYSICAL_OOF"
        ):
            raise ValueError(f"OOF final endpoint is not complete and parent-free: {stage_id}")
        fits[stage_id] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
            "identity_sha256": identity.get("identity_sha256"),
            "training_cache_sha256": identity.get("training_cache_sha256"),
            "training_checkpoint_parents": {},
        }
    transitions = {
        "E_CTX_MATCHED": build_deployment_transition(
            stage_id="E_CTX_MATCHED",
            training_stitch_path=run / "final/oof/PHASE17_RAW_EVENT.authority.json",
            training_normalizer_sha256=sha256(run / "final/oof/event_normalizer.json"),
            deployment_dependencies={
                key: deployment_dependencies[key] for key in ("E_A5", "E_C2F", "PAIR_E")
            },
        ),
        "R_CTX": build_deployment_transition(
            stage_id="R_CTX",
            training_stitch_path=run / "final/oof/PHASE17_RAW_RGB.authority.json",
            training_normalizer_sha256=sha256(run / "final/oof/rgb_normalizer.json"),
            deployment_dependencies={
                key: deployment_dependencies[key] for key in ("R_A5", "R_C2F", "PAIR_R")
            },
        ),
        "PHASE_ONLY_EVENT_S7_TRAIN40": build_deployment_transition(
            stage_id="PHASE_ONLY_EVENT_S7_TRAIN40",
            training_stitch_path=run / "final/oof/PHASE20_CONTEXT.authority.json",
            training_normalizer_sha256=None,
            deployment_dependencies={
                key: deployment_dependencies[key] for key in ("E_CTX_MATCHED", "R_CTX")
            },
        ),
    }
    value: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "status": "SEALED_NOT_EVALUATED",
        "candidate": "PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF",
        "recipe_path": str(recipe_path),
        "recipe_sha256": recipe.recipe_sha256,
        "training_feature_ancestry": transitions,
        "deployment_dependencies": dict(sorted(deployment_dependencies.items())),
        "ancestry_domains_are_distinct": True,
        "trained_endpoints": fits,
        "normalizers": {
            name: sha256(run / f"final/oof/{name}_normalizer.json") for name in ("event", "rgb")
        },
        "scope": {
            "private_targets_read": False,
            "test_predictions_computed": False,
            "external_submission": False,
            "official_score_available": False,
        },
    }
    value["seal_identity_sha256"] = canonical_sha256(value)
    output = run / "final" / "MODEL_SEAL.json"
    _write_once(output, value)
    return value


def deployment_dependency_hashes(run: Path) -> dict[str, str]:
    """Resolve the exact full-fit inference graph from admitted checkpoint receipts."""
    run = run.resolve(strict=True)
    config = _config(run)
    recipe = load_recipe(cast(Path, config["recipe"]))
    result = {name: recipe.external_parent_sha256[name] for name in ("E_A5", "E_C2F")}
    rgb_locations = {
        "R_A5": cast(Path, config["active_rgb_runtime"]) / "fits/R_A5",
        "R_C2F": cast(Path, config["active_rgb_runtime"]) / "fits/R_C2F",
    }
    source = load_all_train_source(cast(Path, config["rgb_source"]), cast(Path, config["teacher"]))
    try:
        for name, directory in rgb_locations.items():
            producer_recipe = build_rgb_recipe(
                cast(Path, config["config"]),
                name,
                cast(Path, config["rgb_source"]),
                source,
            )
            model = load_rgb_endpoint(directory, producer_recipe, device="cpu")
            del model
            result[name] = sha256(directory / "checkpoint_last.pt")
    finally:
        source.close()

    def cached(public_name: str, stage_id: str, expected_parents: Mapping[str, str]) -> str:
        directory = run / "final/fits" / stage_id
        receipt = _read((directory / "CHECKPOINT_RECEIPT.json").resolve(strict=True))
        checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = payload.get("identity", {})
        claimed = identity.get("identity_sha256")
        unhashed = dict(identity)
        unhashed.pop("identity_sha256", None)
        declared = stage(recipe, stage_id)
        cache = (
            run / "cache/PAIR_E_MATCHED.npz"
            if stage_id == "PAIR_E_MATCHED"
            else run / "final/cache/PAIR_R.npz"
        )
        verify_execution_binding(identity)
        if (
            receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("scientific_endpoint") is not True
            or receipt.get("complete_state") is not True
            or receipt.get("fit_id") != stage_id
            or int(receipt.get("completed_updates", -1)) != declared.updates
            or receipt.get("checkpoint_sha256") != sha256(checkpoint)
            or receipt.get("identity_sha256") != claimed
            or payload.get("identity_sha256") != claimed
            or canonical_sha256(unhashed) != claimed
            or identity.get("schema") != "final_refit_cached_stage_identity_v1"
            or identity.get("fit_id") != stage_id
            or identity.get("kind") != declared.kind
            or identity.get("recipe_sha256") != recipe.recipe_sha256
            or identity.get("training_cache_sha256") != sha256(cache)
            or identity.get("parent_checkpoint_sha256") != dict(expected_parents)
        ):
            raise ValueError(f"Deployment cached endpoint differs: {public_name}")
        return sha256(checkpoint)

    result["PAIR_E"] = cached("PAIR_E", "PAIR_E_MATCHED", {"E_A5": result["E_A5"]})
    result["PAIR_R"] = cached("PAIR_R", "PAIR_R", {"R_A5": result["R_A5"]})
    locations = {
        "E_CTX_MATCHED": run / "final/fits/E_CTX_MATCHED",
        "R_CTX": run / "final/fits/R_CTX",
    }
    for name, directory in locations.items():
        receipt = _read((directory / "CHECKPOINT_RECEIPT.json").resolve(strict=True))
        checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("scientific_endpoint") is not True
            or receipt.get("checkpoint_sha256") != sha256(checkpoint)
        ):
            raise ValueError(f"Deployment endpoint is incomplete or changed: {name}")
        result[name] = sha256(checkpoint)
    return result


def export_scientific_head(run: Path) -> dict[str, Any]:
    """Export only the sealed PHASE_ONLY head; no end-to-end latency is claimed."""
    run = run.resolve()
    seal_path = run / "final/MODEL_SEAL.json"
    seal = _read(seal_path.resolve(strict=True))
    if seal.get("schema") != SEAL_SCHEMA or seal.get("status") != "SEALED_NOT_EVALUATED":
        raise ValueError("Scientific OOF model is not sealed")
    component = seal["trained_endpoints"]["PHASE_ONLY_EVENT_S7_TRAIN40"]
    checkpoint = Path(component["checkpoint_path"]).resolve(strict=True)
    if sha256(checkpoint) != component["checkpoint_sha256"]:
        raise ValueError("Sealed PHASE_ONLY checkpoint changed")
    from .training import PhaseOnlyFinal

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PhaseOnlyFinal().eval()
    model.load_state_dict(payload["model_state_dict"])
    output = run / "final" / "PHASE_ONLY_HEAD.ts"

    def verify_export(path: Path) -> float:
        scripted_model = torch.jit.load(str(path), map_location="cpu").eval()
        scripted_state = scripted_model.state_dict()
        eager_state = model.state_dict()
        if set(scripted_state) != set(eager_state) or any(
            not torch.equal(scripted_state[name], eager_state[name]) for name in eager_state
        ):
            raise ValueError("TorchScript export weights differ from the sealed head")
        maximum = 0.0
        for batch in (1, 7):
            features = torch.linspace(-0.25, 0.25, batch * 340).reshape(batch, 340)
            widths = torch.linspace(-0.1, 0.2, batch * 2).reshape(batch, 2)
            with torch.inference_mode():
                eager = model(features, widths)
                scripted_output = scripted_model(features, widths)
            if set(eager) != set(scripted_output):
                raise ValueError("TorchScript output fields differ from the sealed head")
            for name in eager:
                difference = float((eager[name] - scripted_output[name]).abs().max())
                maximum = max(maximum, difference)
        if maximum > 1e-6:
            raise ValueError(f"TorchScript parity differs by {maximum}")
        return maximum

    if output.exists():
        maximum = verify_export(output)
        sidecar = output.with_suffix(".json")
        if not sidecar.exists():
            existing = None
        else:
            existing = _read(sidecar.resolve(strict=True))
        if existing is None:
            existing = {
                "schema": EXPORT_SCHEMA,
                "status": "EXPORTED_NOT_EVALUATED",
                "model_seal_sha256": sha256(seal_path),
                "checkpoint_sha256": sha256(checkpoint),
                "output": str(output.resolve()),
                "output_sha256": sha256(output),
                "export_level": "PHASE_ONLY_HEAD_OVER_PHYSICAL_STATE340",
                "end_to_end_sensor_runtime_exported": False,
                "state_dict_exact": True,
                "parity_batches": [1, 7],
                "parity_max_abs": maximum,
                "parity_tolerance": 1e-6,
            }
            existing["receipt_sha256"] = canonical_sha256(existing)
            _write_once(sidecar, existing)
        if (
            existing.get("schema") != EXPORT_SCHEMA
            or existing.get("status") != "EXPORTED_NOT_EVALUATED"
            or existing.get("model_seal_sha256") != sha256(seal_path)
            or existing.get("checkpoint_sha256") != sha256(checkpoint)
            or existing.get("output_sha256") != sha256(output)
            or existing.get("state_dict_exact") is not True
            or existing.get("parity_batches") != [1, 7]
            or float(existing.get("parity_max_abs", float("inf"))) != maximum
            or float(existing.get("parity_tolerance", -1)) != 1e-6
            or existing.get("receipt_sha256")
            != canonical_sha256(
                {key: value for key, value in existing.items() if key != "receipt_sha256"}
            )
        ):
            raise ValueError("Existing scientific head export changed")
        return existing
    scripted = torch.jit.script(model)
    temporary = output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.ts")
    torch.jit.save(scripted, temporary)
    maximum = verify_export(temporary)
    os.replace(temporary, output)
    receipt = {
        "schema": EXPORT_SCHEMA,
        "status": "EXPORTED_NOT_EVALUATED",
        "model_seal_sha256": sha256(seal_path),
        "checkpoint_sha256": sha256(checkpoint),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "export_level": "PHASE_ONLY_HEAD_OVER_PHYSICAL_STATE340",
        "end_to_end_sensor_runtime_exported": False,
        "state_dict_exact": True,
        "parity_batches": [1, 7],
        "parity_max_abs": maximum,
        "parity_tolerance": 1e-6,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_once(output.with_suffix(".json"), receipt)
    return receipt


def _scientific_seal(path: Path) -> tuple[dict[str, Any], str]:
    seal = _read(path.resolve(strict=True))
    unhashed = dict(seal)
    claimed = unhashed.pop("seal_identity_sha256", None)
    recipe_path = Path(str(seal.get("recipe_path", ""))).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status") != "SEALED_NOT_EVALUATED"
        or seal.get("candidate") != "PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF"
        or seal.get("ancestry_domains_are_distinct") is not True
        or seal.get("scope", {}).get("private_targets_read") is not False
        or claimed != canonical_sha256(unhashed)
        or seal.get("recipe_sha256") != recipe.recipe_sha256
    ):
        raise ValueError("Test12 requires the sealed scientific OOF final chain")
    return seal, sha256(path)


class ScientificOOFFinalPredictor(FinalV16Train40Predictor):
    """Sensor-to-TTC adapter for OOF-trained heads and full-fit deployment parents."""

    def __init__(
        self,
        *,
        run: Path,
        model_seal_path: Path,
        inputs: Mapping[str, Any],
        device: str,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        run = run.resolve(strict=True)
        config = _config(run)
        self.device = torch.device(device)
        self.work_directory = run / "final" / "test12"
        self.inputs = inputs
        self.seal, self.seal_sha = _scientific_seal(model_seal_path)
        self.recipe = load_recipe(cast(Path, config["recipe"]))
        if self.seal.get("recipe_sha256") != self.recipe.recipe_sha256:
            raise ValueError("Scientific seal and final recipe differ")
        dependencies = cast(dict[str, str], self.seal["deployment_dependencies"])
        endpoints = cast(dict[str, dict[str, Any]], self.seal["trained_endpoints"])
        self.models = {}
        protocol = _read(cast(Path, config["v12_protocol"]))
        for arm, seal_name, runtime_name in (
            ("a5", "E_A5", "E_A5_MATCHED"),
            ("c2f", "E_C2F", "E_C2F_MATCHED"),
        ):
            model, identity = load_v12_event_producer(cast(Path, config["v12_run"]), protocol, arm)
            if identity["checkpoint_sha256"] != dependencies[seal_name]:
                raise ValueError(f"Sealed event producer differs: {seal_name}")
            self.models[runtime_name] = model.to(self.device).eval().requires_grad_(False)
        source = load_all_train_source(
            cast(Path, config["rgb_source"]), cast(Path, config["teacher"])
        )
        try:
            for fit_id in ("R_A5", "R_C2F"):
                producer_recipe = build_rgb_recipe(
                    cast(Path, config["config"]),
                    fit_id,
                    cast(Path, config["rgb_source"]),
                    source,
                )
                fit = cast(Path, config["active_rgb_runtime"]) / "fits" / fit_id
                checkpoint = fit / "checkpoint_last.pt"
                if sha256(checkpoint) != dependencies[fit_id]:
                    raise ValueError(f"Sealed RGB producer differs: {fit_id}")
                self.models[fit_id] = load_rgb_endpoint(
                    fit, producer_recipe, device=str(self.device)
                )
        finally:
            source.close()
        for fit_id, dependency_name in (
            ("PAIR_E_MATCHED", "PAIR_E"),
            ("PAIR_R", "PAIR_R"),
        ):
            fit = run / "final" / "fits" / fit_id
            checkpoint = fit / "checkpoint_last.pt"
            if sha256(checkpoint) != dependencies[dependency_name]:
                raise ValueError(f"Sealed deployment endpoint differs: {fit_id}")
            self.models[fit_id] = load_cached_endpoint(fit, stage(self.recipe, fit_id), self.device)
        for fit_id in ("E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40"):
            fit = run / "final" / "fits" / fit_id
            checkpoint = fit / "checkpoint_last.pt"
            if sha256(checkpoint) != endpoints[fit_id]["checkpoint_sha256"]:
                raise ValueError(f"Sealed OOF-trained endpoint differs: {fit_id}")
            self.models[fit_id] = load_cached_endpoint(
                fit, _oof_stage(self.recipe, fit_id), self.device
            )
        self.normalizers = {}
        for modality in ("event", "rgb"):
            path = run / "final" / "oof" / f"{modality}_normalizer.json"
            if sha256(path) != self.seal["normalizers"][modality]:
                raise ValueError(f"Sealed {modality} OOF normalizer changed")
            payload = _read(path)
            self.normalizers[modality] = (
                torch.tensor(payload["mean"], dtype=torch.float32, device=self.device),
                torch.tensor(payload["scale"], dtype=torch.float32, device=self.device),
            )
        self.reader = TarFrameReader(Path(str(inputs["media_root"])), max_open=2)
        self.event_values = {}
        self.rgb_values = {}


def run_scientific_test12(
    *,
    run: Path,
    predictor: Test12Predictor,
    predictions_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Run/resume the exact trusted public inventory under the scientific model seal."""
    run = run.resolve(strict=True)
    config = _config(run)
    inputs_path = cast(Path, config["test12_inputs"])
    official_path = cast(Path, config["official_test_inputs"])
    seal_path = run / "final" / "MODEL_SEAL.json"
    validated = validate_test12_inputs(
        inputs_path,
        official_test_inputs_path=official_path,
        expected_inputs_sha256=TRUSTED_INPUTS_SHA256,
        expected_official_test_inputs_sha256=TRUSTED_OFFICIAL_TEST_INPUTS_SHA256,
        expected_model_input_freeze_sha256=TRUSTED_MODEL_INPUT_FREEZE_SHA256,
        expected_event_input_freeze_sha256=TRUSTED_EVENT_INPUT_FREEZE_SHA256,
        model_input_freeze_path=cast(Path, config["model_input_freeze"]),
        event_input_freeze_path=cast(Path, config["event_input_freeze"]),
    )
    _, seal_sha = _scientific_seal(seal_path)
    binding = canonical_sha256(
        {"model_seal_sha256": seal_sha, "inputs_sha256": validated["inputs_sha256"]}
    )
    fragments = PredictionFragments(run / "final" / "test12" / "fragments", binding)
    from .scientific_oof_pipeline import make_guard

    guard = make_guard(run)
    inputs = cast(dict[str, Any], validated["inputs"])
    by_token = {str(row["sample_token"]): row for row in inputs["queries"]}
    aliases = cast(dict[str, str], inputs["physical_aliases"])
    physical: dict[str, float] = {}
    try:
        for token in validated["canonical_tokens"]:
            guard()
            value = fragments.read(token)
            if value is None:
                value = float(predictor.predict(by_token[token], inputs))
                fragments.write(token, value)
            physical[token] = value
    finally:
        predictor.close()
    records = [
        {
            "sample_token": token,
            "prediction": physical[str(aliases[token])],
            "rgb_available": bool(by_token[token]["rgb_history"]),
            "fallback_source": ("PHASE_ONLY" if by_token[token]["rgb_history"] else "E_CTX"),
        }
        for token in validated["tokens"]
    ]
    frame = pd.DataFrame.from_records(records)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions_output.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    if predictions_output.exists() and predictions_output.read_bytes() != temporary.read_bytes():
        temporary.unlink()
        raise ValueError("Resumed scientific predictions differ")
    if predictions_output.exists():
        temporary.unlink()
    else:
        os.replace(temporary, predictions_output)
    receipt = {
        "schema": PREDICTION_SCHEMA,
        "status": "COMPLETE",
        "candidate": self_candidate(),
        "model_seal_sha256": seal_sha,
        "inputs_sha256": validated["inputs_sha256"],
        "predictions_sha256": sha256(predictions_output),
        "population": OFFICIAL_TEST_QUERIES,
        "physical_queries": len(validated["canonical_tokens"]),
        "computed_under_seal": True,
        "private_targets_read": False,
        "external_submission": False,
        "fallback_count": validated["fallback_count"],
        "run_binding_sha256": binding,
    }
    _write_once(receipt_output, receipt)
    return receipt


def self_candidate() -> str:
    """Canonical public candidate identity used by seal and prediction receipts."""
    return "PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF"


def package_scientific_predictions(run: Path) -> dict[str, Any]:
    """Package the already sealed 6,762 predictions without submitting them."""
    run = run.resolve(strict=True)
    config = _config(run)
    predictions = run / "final/test12/predictions.csv"
    prediction_receipt = _read((run / "final/test12/PREDICTION_RECEIPT.json").resolve(strict=True))
    seal_path = run / "final/MODEL_SEAL.json"
    _, seal_sha = _scientific_seal(seal_path)
    if (
        prediction_receipt.get("status") != "COMPLETE"
        or prediction_receipt.get("model_seal_sha256") != seal_sha
        or prediction_receipt.get("predictions_sha256") != sha256(predictions)
    ):
        raise ValueError("Scientific predictions are incomplete or detached from the seal")
    official = pd.read_parquet(cast(Path, config["official_test_inputs"]), columns=["sample_token"])
    tokens = official.sample_token.astype(str).tolist()
    frame = pd.read_csv(predictions)
    if (
        len(tokens) != OFFICIAL_TEST_QUERIES
        or len(set(tokens)) != OFFICIAL_TEST_QUERIES
        or frame.sample_token.astype(str).tolist() != tokens
        or not np.isfinite(frame.prediction.to_numpy(np.float64)).all()
    ):
        raise ValueError("Scientific predictions do not exactly cover official Test12 order")
    submission = {
        "meta": {"format": "garlttc_prediction_v1"},
        "results": {
            token: {"ttc": float(value)}
            for token, value in zip(tokens, frame.prediction, strict=True)
        },
    }
    output = run / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip"

    def verify_zip(path: Path) -> None:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.namelist() != ["submission.json"]:
                raise ValueError("Scientific package must contain one root submission.json")
            decoded = json.loads(archive.read("submission.json").decode("utf-8"))
        if decoded != submission:
            raise ValueError("Scientific package content differs from current predictions")

    if output.exists():
        verify_zip(output)
        existing = _read(output.with_suffix(".json").resolve(strict=True))
        if (
            existing.get("schema") != "scientific_oof_test12_package_v1"
            or existing.get("status") != "PACKAGED_NOT_SUBMITTED"
            or existing.get("candidate") != self_candidate()
            or existing.get("model_seal_sha256") != seal_sha
            or existing.get("zip_sha256") != sha256(output)
            or int(existing.get("population", -1)) != len(tokens)
            or existing.get("prediction_receipt_sha256")
            != sha256(run / "final/test12/PREDICTION_RECEIPT.json")
            or existing.get("private_targets_read") is not False
            or existing.get("external_submission") is not False
            or existing.get("receipt_sha256")
            != canonical_sha256(
                {key: value for key, value in existing.items() if key != "receipt_sha256"}
            )
        ):
            raise ValueError("Existing scientific submission package changed")
        return existing
    payload = json.dumps(submission, separators=(",", ":"), allow_nan=False).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.zip")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", payload)
    verify_zip(temporary)
    os.replace(temporary, output)
    receipt = {
        "schema": "scientific_oof_test12_package_v1",
        "status": "PACKAGED_NOT_SUBMITTED",
        "candidate": self_candidate(),
        "model_seal_sha256": seal_sha,
        "prediction_receipt_sha256": sha256(run / "final/test12/PREDICTION_RECEIPT.json"),
        "population": len(tokens),
        "zip_path": str(output),
        "zip_sha256": sha256(output),
        "private_targets_read": False,
        "external_submission": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_once(output.with_suffix(".json"), receipt)
    return receipt


def run_all(run: Path, *, max_updates: int = 100) -> dict[str, Any]:
    """Complete every recoverable final stage through a local, unsubmitted Test12 ZIP."""
    run = run.resolve(strict=True)
    prepare_oof_training(run)
    prepare_deployment_rgb_pair(run)
    while True:
        progress = fit_next_oof_stage(run, max_updates=max_updates)
        if progress.get("all_head_endpoints_complete") is True:
            break
    seal_scientific_final(run, deployment_dependencies=deployment_dependency_hashes(run))
    export_scientific_head(run)
    config = _config(run)
    validated = validate_test12_inputs(
        cast(Path, config["test12_inputs"]),
        official_test_inputs_path=cast(Path, config["official_test_inputs"]),
        model_input_freeze_path=cast(Path, config["model_input_freeze"]),
        event_input_freeze_path=cast(Path, config["event_input_freeze"]),
    )
    predictor = ScientificOOFFinalPredictor(
        run=run,
        model_seal_path=run / "final/MODEL_SEAL.json",
        inputs=cast(dict[str, Any], validated["inputs"]),
        device=cast(str, config["device"]),
    )
    run_scientific_test12(
        run=run,
        predictor=predictor,
        predictions_output=run / "final/test12/predictions.csv",
        receipt_output=run / "final/test12/PREDICTION_RECEIPT.json",
    )
    package_scientific_predictions(run)
    return _complete_final_receipt(
        run,
        {
            "model_seal": run / "final/MODEL_SEAL.json",
            "head_export": run / "final/PHASE_ONLY_HEAD.ts",
            "head_export_receipt": run / "final/PHASE_ONLY_HEAD.json",
            "predictions": run / "final/test12/predictions.csv",
            "prediction_receipt": run / "final/test12/PREDICTION_RECEIPT.json",
            "submission_zip": run / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip",
            "package_receipt": run / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json",
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument(
        "--action",
        choices=("prepare", "train", "seal", "export", "test12"),
        default="train",
    )
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare_oof_training(args.run)
        return 0
    if args.action == "seal":
        receipt = seal_scientific_final(
            args.run, deployment_dependencies=deployment_dependency_hashes(args.run)
        )
    elif args.action == "export":
        receipt = export_scientific_head(args.run)
    elif args.action == "test12":
        config = _config(args.run)
        validated = validate_test12_inputs(
            cast(Path, config["test12_inputs"]),
            official_test_inputs_path=cast(Path, config["official_test_inputs"]),
            model_input_freeze_path=cast(Path, config["model_input_freeze"]),
            event_input_freeze_path=cast(Path, config["event_input_freeze"]),
        )
        predictor = ScientificOOFFinalPredictor(
            run=args.run,
            model_seal_path=args.run / "final/MODEL_SEAL.json",
            inputs=cast(dict[str, Any], validated["inputs"]),
            device=cast(str, config["device"]),
        )
        receipt = run_scientific_test12(
            run=args.run,
            predictor=predictor,
            predictions_output=args.run / "final/test12/predictions.csv",
            receipt_output=args.run / "final/test12/PREDICTION_RECEIPT.json",
        )
        receipt = package_scientific_predictions(args.run)
    else:
        try:
            receipt = run_all(args.run, max_updates=args.max_updates)
        except Exception as error:
            from .scientific_oof_pipeline import ResourcePause

            if not isinstance(error, ResourcePause):
                raise
            receipt = _write_final_receipt(args.run, "FINAL", {}, 0)
            receipt["status"] = "PAUSED_RESOURCE"
            receipt["manual_stop"] = bool(error.manual)
            receipt["resources"] = error.resources
            receipt["receipt_sha256"] = canonical_sha256(
                {key: item for key, item in receipt.items() if key != "receipt_sha256"}
            )
            atomic_write_json(args.run / "final/FINAL_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    successful = {
        "COMPLETE",
        "SEALED_NOT_EVALUATED",
        "EXPORTED_NOT_EVALUATED",
        "PACKAGED_NOT_SUBMITTED",
    }
    return 0 if receipt.get("status") in successful else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPORT_SCHEMA",
    "SCHEMA",
    "SEAL_SCHEMA",
    "export_scientific_head",
    "deployment_dependency_hashes",
    "fit_next_oof_stage",
    "prepare_oof_training",
    "prepare_deployment_rgb_pair",
    "package_scientific_predictions",
    "run_scientific_test12",
    "run_all",
    "seal_scientific_final",
    "ScientificOOFFinalPredictor",
]
