"""Seal, export, and locally package the completed final TRAIN40 model.

This module exports the PHASE_ONLY readout over already assembled 340-D raw
states.  It does not claim end-to-end sensor latency and never submits a ZIP.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional

from operational.sota_evidence.codabench import package_predictions

from .features import PHASE340_SCHEMA_SHA256, validate_cache_receipt
from .producer_plan import SCHEMA as PRODUCER_PLAN_SCHEMA
from .producer_plan import load_plan
from .recipe import FinalRefitRecipe, canonical_sha256, load_recipe, sha256, stage
from .training import PhaseOnlyFinal, load_cached_endpoint

MODEL_SEAL_SCHEMA = "final_refit_complete_model_seal_v1"
EXPORT_SCHEMA = "final_refit_phase_only_torchscript_v1"
PREDICTION_SCHEMA = "final_refit_test12_predictions_v1"
OFFICIAL_TEST_QUERIES = 6_762
HEAD_STAGE = "PHASE_ONLY_EVENT_S7_TRAIN40"
EVENT_PARENTS = {"E_A5", "E_C2F"}
NORMALIZERS = {"rgb", "event"}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _complete_stage(
    name: str,
    receipt_path: Path,
    *,
    recipe: FinalRefitRecipe,
    cache_receipt_path: Path | None,
) -> dict[str, Any]:
    declared = stage(recipe, name)
    receipt = _read(receipt_path)
    checkpoint = Path(str(receipt.get("checkpoint_path", ""))).resolve(strict=True)
    checkpoint_sha256 = sha256(checkpoint)
    if (
        receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
        or receipt.get("fit_id") != name
        or receipt.get("status") != "COMPLETE"
        or receipt.get("scientific_endpoint") is not True
        or int(receipt.get("completed_updates", -1)) != declared.updates
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("complete_state") is not True
        or int(receipt.get("accumulation_index", -1)) != 0
    ):
        raise ValueError(f"Stage {name} is not an immutable complete endpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Stage {name} checkpoint has no immutable training identity")
    identity_without_hash = dict(identity)
    claimed_identity = identity_without_hash.pop("identity_sha256", None)
    common_valid = (
        payload.get("schema") == "rgb_port_producer_checkpoint_v1"
        and payload.get("identity_sha256") == claimed_identity
        and receipt.get("identity_sha256") == claimed_identity
        and claimed_identity == canonical_sha256(identity_without_hash)
        and identity.get("fit_id") == name
    )
    if declared.kind == "rgb_producer":
        contract = Path(str(identity.get("producer_contract_path", ""))).resolve(strict=True)
        descriptor = Path(str(identity.get("source_descriptor_path", ""))).resolve(strict=True)
        plan = load_plan(contract)
        matches = [item for item in plan.stages if item.fit_id == name]
        teacher = identity.get("teacher_policy")
        executed = identity.get("executed_source_sha256")
        common_sources = {
            "operational/final_refit/producers.py",
            "operational/final_refit/rgb.py",
            "operational/rgb_port/train_producers.py",
            "operational/rgb_port_revision/loss.py",
        }
        identity_valid = (
            identity.get("schema") == "final_refit_rgb_producer_identity_v1"
            and int(identity.get("producer_population", -1)) == recipe.train_queries
            and identity.get("population_role") == "TRAIN"
            and identity.get("historical_roles_preserved") == ["P", "H", "V"]
            and identity.get("initialization") == "fresh_seed7_no_event_or_historical_rgb_weights"
            and identity.get("producer_contract_schema") == PRODUCER_PLAN_SCHEMA
            and identity.get("producer_contract_sha256") == sha256(contract)
            and plan.launch_authorized
            and plan.status == "FROZEN_AUTHORIZED_PRODUCER_SCOPE"
            and len(matches) == 1
            and matches[0].updates == declared.updates
            and matches[0].population == recipe.train_queries
            and matches[0].fresh_initialization
            and not matches[0].parent_checkpoints
            and identity.get("source_descriptor_sha256") == sha256(descriptor)
            and plan.input_sha256.get("source_descriptor") == sha256(descriptor)
            and isinstance(identity.get("source_identity"), dict)
            and bool(identity["source_identity"])
            and isinstance(teacher, dict)
            and teacher.get("teacher_manifest_sha256") == plan.input_sha256.get("teacher_manifest")
            and teacher.get("teacher_rows_sha256") == plan.input_sha256.get("train40_rows")
            and isinstance(executed, dict)
            and all(
                executed.get(key) == plan.executed_source_sha256.get(key) for key in common_sources
            )
            and executed.get("configs/rgb_port/producers.json")
            == plan.input_sha256.get("producer_config")
        )
    else:
        parent_checkpoints = identity.get("parent_checkpoint_sha256")
        identity_valid = (
            identity.get("schema") == "final_refit_cached_stage_identity_v1"
            and identity.get("kind") == declared.kind
            and identity.get("recipe_sha256") == recipe.recipe_sha256
            and identity.get("technical") is False
            and isinstance(parent_checkpoints, dict)
            and set(parent_checkpoints) == set(declared.depends_on)
            and isinstance(identity.get("cache_sha256"), str)
            and len(identity["cache_sha256"]) == 64
            and isinstance(identity.get("cache_receipt_sha256"), str)
            and len(identity["cache_receipt_sha256"]) == 64
        )
        if cache_receipt_path is None or declared.cache_schema is None:
            raise ValueError(f"Stage {name} requires its exact training-cache receipt")
        cache_receipt_path = cache_receipt_path.resolve(strict=True)
        cache_receipt = _read(cache_receipt_path)
        cache_path = Path(str(cache_receipt.get("output", ""))).resolve(strict=True)
        validated_cache = validate_cache_receipt(
            cache_path, declared.cache_schema, expected_role="TRAIN"
        )
        identity_valid = (
            identity_valid
            and sha256(cache_path) == identity.get("cache_sha256")
            and sha256(cache_receipt_path) == identity.get("cache_receipt_sha256")
        )
    if declared.kind == "rgb_producer":
        validated_cache = None
    if not common_valid or not identity_valid:
        raise ValueError(f"Stage {name} training identity differs from the final recipe")
    return {
        "stage": name,
        "kind": declared.kind,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256(receipt_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "completed_updates": declared.updates,
        "identity_sha256": receipt.get("identity_sha256"),
        "parent_checkpoint_sha256": dict(identity.get("parent_checkpoint_sha256", {})),
        "cache_sha256": identity.get("cache_sha256"),
        "cache_receipt_sha256": identity.get("cache_receipt_sha256"),
        "producer_contract_sha256": identity.get("producer_contract_sha256"),
        "source_descriptor_sha256": identity.get("source_descriptor_sha256"),
        "source_identity": identity.get("source_identity"),
        "teacher_manifest_sha256": (
            identity.get("teacher_policy", {}).get("teacher_manifest_sha256")
            if isinstance(identity.get("teacher_policy"), dict)
            else None
        ),
        "executed_source_sha256": identity.get("executed_source_sha256"),
        "cache_normalizers": (
            dict(validated_cache.get("normalizers", {})) if validated_cache is not None else {}
        ),
    }


def _event_parent(name: str, receipt_path: Path, expected_sha256: str) -> dict[str, Any]:
    receipt = _read(receipt_path)
    checkpoint = Path(str(receipt.get("checkpoint_path", ""))).resolve(strict=True)
    if (
        receipt.get("schema") != "final_refit_external_parent_admission_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("reuse_authorized") is not True
        or receipt.get("fit_id") != name
        or receipt.get("split_version") != "public_TRAIN40_all_train"
        or receipt.get("checkpoint_sha256") != expected_sha256
        or receipt.get("checkpoint_sha256") != sha256(checkpoint)
        or receipt.get("input_manifest_sha256") is None
        or receipt.get("teacher_manifest_sha256") is None
    ):
        raise ValueError(f"External event parent {name} is not admitted")
    return {
        "stage": name,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256(receipt_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_sha256,
        "split_version": receipt["split_version"],
        "input_manifest_sha256": receipt["input_manifest_sha256"],
        "teacher_manifest_sha256": receipt["teacher_manifest_sha256"],
    }


def _normalizer(name: str, path: Path) -> dict[str, Any]:
    value = _read(path)
    claimed = value.pop("normalizer_payload_sha256", None)
    if (
        name not in NORMALIZERS
        or value.get("schema") != "final_refit_train40_normalizer_v1"
        or value.get("status") != "COMPLETE"
        or value.get("role") != "TRAIN"
        or value.get("modality") != name
        or value.get("targets_read") is not False
        or claimed != canonical_sha256(value)
        or np.asarray(value.get("mean", [])).shape != (17,)
        or np.asarray(value.get("scale", [])).shape != (17,)
        or not np.isfinite(np.asarray(value["mean"], dtype=float)).all()
        or not (np.asarray(value["scale"], dtype=float) > 0).all()
        or not isinstance(value.get("parents"), dict)
        or not value["parents"]
    ):
        raise ValueError(f"{name} normalizer is not a complete TRAIN-only artifact")
    return {
        "modality": name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "payload_sha256": claimed,
        "rows": int(value.get("rows", -1)),
        "parents": dict(value["parents"]),
    }


def seal_final_chain(
    *,
    recipe_path: Path,
    stage_receipts: Mapping[str, Path],
    cache_receipts: Mapping[str, Path],
    event_parent_receipts: Mapping[str, Path],
    normalizers: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    """Rehash and seal the complete 7-stage chain plus two reused event parents."""
    recipe_path = recipe_path.resolve(strict=True)
    recipe = load_recipe(recipe_path)
    expected_stages = {item.stage_id for item in recipe.stages}
    if set(stage_receipts) != expected_stages:
        raise ValueError("Stage receipt set differs from the seven-stage final recipe")
    cached_stages = {item.stage_id for item in recipe.stages if item.cache_schema is not None}
    if set(cache_receipts) != cached_stages:
        raise ValueError("Cache receipt set differs from the five cached final stages")
    if set(event_parent_receipts) != EVENT_PARENTS:
        raise ValueError("Event parent receipt set must be E_A5 and E_C2F")
    if set(normalizers) != NORMALIZERS:
        raise ValueError("Exactly the rgb and event normalizers are required")
    stages = [
        _complete_stage(
            item.stage_id,
            Path(stage_receipts[item.stage_id]),
            recipe=recipe,
            cache_receipt_path=(
                Path(cache_receipts[item.stage_id]) if item.stage_id in cache_receipts else None
            ),
        )
        for item in recipe.stages
    ]
    parents = [
        _event_parent(
            name,
            Path(event_parent_receipts[name]),
            recipe.external_parent_sha256[name],
        )
        for name in sorted(EVENT_PARENTS)
    ]
    sealed_normalizers = [
        _normalizer(name, Path(normalizers[name])) for name in sorted(NORMALIZERS)
    ]
    checkpoint_hashes = {
        **{item["stage"]: item["checkpoint_sha256"] for item in stages},
        **{item["stage"]: item["checkpoint_sha256"] for item in parents},
    }
    for item in stages:
        declared = stage(recipe, str(item["stage"]))
        expected = {name: checkpoint_hashes[name] for name in declared.depends_on}
        if item["parent_checkpoint_sha256"] != expected:
            raise ValueError(f"Stage {item['stage']} parent checkpoint chain differs")
    rgb_stages = [item for item in stages if item["kind"] == "rgb_producer"]
    for field in (
        "producer_contract_sha256",
        "source_descriptor_sha256",
        "source_identity",
        "teacher_manifest_sha256",
        "executed_source_sha256",
    ):
        if len({canonical_sha256(item[field]) for item in rgb_stages}) != 1:
            raise ValueError(f"RGB producer {field} bindings differ")
    expected_normalizer_parents = {
        "event": {
            name: checkpoint_hashes[name] for name in stage(recipe, "E_CTX_MATCHED").depends_on
        },
        "rgb": {name: checkpoint_hashes[name] for name in stage(recipe, "R_CTX").depends_on},
    }
    for item in sealed_normalizers:
        if item["parents"] != expected_normalizer_parents[item["modality"]]:
            raise ValueError(f"{item['modality']} normalizer parent chain differs")
    normalizer_hashes = {item["modality"]: item["sha256"] for item in sealed_normalizers}
    expected_cache_normalizers = {
        "PAIR_E_MATCHED": {},
        "PAIR_R": {},
        "E_CTX_MATCHED": {"event": normalizer_hashes["event"]},
        "R_CTX": {"rgb": normalizer_hashes["rgb"]},
        HEAD_STAGE: normalizer_hashes,
    }
    for item in stages:
        if item["stage"] in expected_cache_normalizers and (
            item["cache_normalizers"] != expected_cache_normalizers[item["stage"]]
        ):
            raise ValueError(f"Stage {item['stage']} cache normalizers differ")
    value: dict[str, Any] = {
        "schema": MODEL_SEAL_SCHEMA,
        "status": "SEALED_NOT_EVALUATED",
        "candidate": "PHASE_ONLY_EVENT_S7_TRAIN40",
        "recipe_path": str(recipe_path),
        "recipe_file_sha256": sha256(recipe_path),
        "recipe_sha256": recipe.recipe_sha256,
        "population": {"train_queries": recipe.train_queries, "train_groups": recipe.train_groups},
        "components": {
            "trained_stages": stages,
            "external_event_parents": parents,
            "normalizers": sealed_normalizers,
            "counts": {"trained_stages": 7, "external_event_parents": 2, "normalizers": 2},
        },
        "head_input_contract": {
            "raw_state": {"dtype": "float32", "shape": ["B", 340]},
            "rgb_width_logits": {"dtype": "float32", "shape": ["B", 2]},
            "feature_schema_sha256": PHASE340_SCHEMA_SHA256,
            "modality_order": "rgb_then_event",
            "selected_columns": [[160, 170], [330, 340]],
            "output_columns": ["ttc", "point_phase", "q10_phase", "q90_phase"],
        },
        "scope": {
            "export_level": "PHASE_ONLY_HEAD_OVER_FROZEN_RAW_STATE340",
            "end_to_end_sensor_runtime_exported": False,
            "end_to_end_sensor_latency_claimed": False,
            "test_predictions_computed": False,
            "private_targets_read": False,
            "external_submission": False,
        },
    }
    value["seal_identity_sha256"] = canonical_sha256(value)
    _atomic_json(output.resolve(), value)
    return value


class _PhaseOnlyExportWrapper(nn.Module):
    """TorchScript-friendly fixed tensor interface for the final readout."""

    def __init__(self, head: PhaseOnlyFinal) -> None:
        super().__init__()
        self.mlp = head.mlp
        self.output = head.output

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> Tensor:
        selected = torch.cat((features[:, 160:170], features[:, 330:340]), dim=-1).clone()
        selected[:, :5] = selected[:, :5] / 0.03
        selected[:, 10:15] = selected[:, 10:15] / 0.03
        correction = self.output(self.mlp(selected))
        location = features[:, 160] + 0.03 * correction[:, 0]
        widths = 0.03 * functional.softplus(rgb_width_logits + correction[:, 1:])
        phi = location.clamp(-0.6931461805599453, 8.0)
        q = -torch.expm1(-phi) / 0.1
        sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
        ttc = sign / q.abs().clamp_min(1.0 / 60.0)
        point_phase = -torch.log1p(-0.1 / ttc)
        q10 = (location - widths[:, 0]).clamp(-0.6931461805599453, 8.0)
        q90 = (location + widths[:, 1]).clamp(-0.6931461805599453, 8.0)
        return torch.stack((ttc, point_phase, q10, q90), dim=1)


def _source_outputs(head: PhaseOnlyFinal, features: Tensor, widths: Tensor) -> Tensor:
    value = head(features, widths)
    return torch.stack((value["ttc"], value["point_phase"], value["q10"], value["q90"]), dim=1)


def export_phase_only_head(
    *, recipe_path: Path, fit_directory: Path, model_seal_path: Path, output: Path
) -> dict[str, Any]:
    """Export the complete final head and prove CPU FP32 parity after reload."""
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise FileExistsError(output)
    recipe = load_recipe(recipe_path.resolve(strict=True))
    declared = stage(recipe, HEAD_STAGE)
    seal = _read(model_seal_path.resolve(strict=True))
    if seal.get("schema") != MODEL_SEAL_SCHEMA or seal.get("recipe_sha256") != recipe.recipe_sha256:
        raise ValueError("Model seal does not bind the final recipe")
    seal_sha256 = sha256(model_seal_path)
    head_component = next(
        (item for item in seal["components"]["trained_stages"] if item["stage"] == HEAD_STAGE),
        None,
    )
    receipt_path = fit_directory.resolve(strict=True) / "CHECKPOINT_RECEIPT.json"
    if head_component is None or head_component.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError("Model seal does not bind the requested PHASE_ONLY endpoint")
    head = cast(
        PhaseOnlyFinal,
        load_cached_endpoint(fit_directory.resolve(strict=True), declared, torch.device("cpu")),
    )
    wrapper = _PhaseOnlyExportWrapper(head).eval().requires_grad_(False)
    scripted = torch.jit.script(wrapper)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.pt")
    with temporary.open("xb") as stream:
        torch.jit.save(scripted, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    with output.open("rb") as stream:
        restored = torch.jit.load(stream, map_location="cpu").eval()
    generator = torch.Generator(device="cpu").manual_seed(recipe.seed)
    maximum = 0.0
    with torch.inference_mode():
        for batch in (1, 7):
            features = torch.randn((batch, 340), generator=generator, dtype=torch.float32)
            widths = torch.randn((batch, 2), generator=generator, dtype=torch.float32)
            expected = _source_outputs(head, features, widths)
            observed = restored(features, widths)
            if expected.shape != observed.shape or not torch.isfinite(observed).all():
                raise ValueError("Exported PHASE_ONLY output contract differs")
            maximum = max(maximum, float((expected - observed).abs().max()))
    if maximum > 1e-6:
        raise ValueError(f"TorchScript parity failed: max absolute difference {maximum}")
    receipt: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "status": "EXPORTED_PARITY_PASS",
        "candidate": HEAD_STAGE,
        "model_seal_sha256": seal_sha256,
        "torchscript_path": str(output.resolve()),
        "torchscript_sha256": sha256(output),
        "head_checkpoint_sha256": head_component["checkpoint_sha256"],
        "input_contract": seal["head_input_contract"],
        "parity": {
            "device": "cpu",
            "dtype": "float32",
            "batches": [1, 7],
            "atol": 1e-6,
            "rtol": 0.0,
            "max_abs_difference": maximum,
        },
        "end_to_end_sensor_runtime_exported": False,
        "latency_claimed": False,
        "predictions_computed": False,
        "external_submission": False,
    }
    _atomic_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def package_final_predictions(
    *,
    inputs_path: Path,
    predictions_path: Path,
    prediction_receipt_path: Path,
    model_seal_path: Path,
    output: Path,
    receipt_output: Path | None = None,
) -> dict[str, Any]:
    """Validate seal-bound real test12 predictions and create a local ZIP only."""
    inputs = _read(inputs_path.resolve(strict=True))
    seal = _read(model_seal_path.resolve(strict=True))
    prediction_receipt = _read(prediction_receipt_path.resolve(strict=True))
    if seal.get("schema") != MODEL_SEAL_SCHEMA or seal.get("status") != "SEALED_NOT_EVALUATED":
        raise ValueError("Final model seal is missing or invalid")
    queries = inputs.get("queries")
    aliases = inputs.get("physical_aliases", {})
    if not isinstance(queries, list) or not isinstance(aliases, dict):
        raise ValueError("Frozen official INPUTS has no query/alias contract")
    tokens = [str(item.get("sample_token", "")) for item in queries]
    if len(tokens) != OFFICIAL_TEST_QUERIES or len(set(tokens)) != OFFICIAL_TEST_QUERIES:
        raise ValueError("Frozen official INPUTS must contain exactly 6,762 unique tokens")
    seal_sha256 = sha256(model_seal_path)
    predictions_sha256 = sha256(predictions_path.resolve(strict=True))
    if (
        prediction_receipt.get("schema") != PREDICTION_SCHEMA
        or prediction_receipt.get("status") != "COMPLETE"
        or prediction_receipt.get("candidate") != HEAD_STAGE
        or prediction_receipt.get("model_seal_sha256") != seal_sha256
        or prediction_receipt.get("inputs_sha256") != sha256(inputs_path)
        or prediction_receipt.get("predictions_sha256") != predictions_sha256
        or int(prediction_receipt.get("population", -1)) != OFFICIAL_TEST_QUERIES
        or prediction_receipt.get("computed_under_seal") is not True
        or prediction_receipt.get("private_targets_read") is not False
        or prediction_receipt.get("external_submission") is not False
        or prediction_receipt.get("fallback_policy") != "E_CTX_WHEN_RGB_HISTORY_UNAVAILABLE"
    ):
        raise ValueError("Prediction receipt does not prove a complete seal-bound inference")
    frame = pd.read_csv(predictions_path)
    required = {"sample_token", "prediction", "rgb_available", "fallback_source"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Prediction columns differ: missing {missing}")
    if frame["sample_token"].duplicated().any() or set(frame["sample_token"].astype(str)) != set(
        tokens
    ):
        raise ValueError("Prediction tokens differ from frozen official INPUTS")
    frame = frame.copy()
    frame["sample_token"] = frame["sample_token"].astype(str)
    lookup = frame.set_index("sample_token", drop=False)
    missing_rgb = 0
    for query in queries:
        token = str(query["sample_token"])
        expected_rgb = bool(query.get("rgb_history"))
        row = lookup.loc[token]
        if bool(row["rgb_available"]) != expected_rgb:
            raise ValueError("Prediction RGB availability differs from frozen INPUTS")
        expected_source = "PHASE_ONLY" if expected_rgb else "E_CTX"
        if str(row["fallback_source"]) != expected_source:
            raise ValueError("Prediction fallback source differs from the frozen policy")
        missing_rgb += int(not expected_rgb)
    if int(prediction_receipt.get("fallback_count", -1)) != missing_rgb:
        raise ValueError("Prediction receipt fallback count differs")
    for alias, canonical in aliases.items():
        if alias not in lookup.index or canonical not in lookup.index:
            raise ValueError("Physical alias references an unknown official token")
        if float(lookup.loc[alias, "prediction"]) != float(lookup.loc[canonical, "prediction"]):
            raise ValueError("Physical aliases have different final predictions")
    package = package_predictions(tokens, frame, model_seal_path, output)
    package.update(
        schema="final_refit_codabench_package_v1",
        candidate=HEAD_STAGE,
        model_seal_sha256=seal_sha256,
        inputs_sha256=sha256(inputs_path),
        predictions_sha256=predictions_sha256,
        prediction_receipt_sha256=sha256(prediction_receipt_path),
        aliases_verified=len(aliases),
        fallback_count=missing_rgb,
        private_labels_read=False,
        external_submission=False,
        official_score_available=False,
        official_sota=False,
    )
    if receipt_output is not None:
        _atomic_json(receipt_output.resolve(), package)
    return package


def _pairs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"Expected unique NAME=PATH, got {value!r}")
        result[name] = Path(raw).resolve(strict=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one local export operation without inference or submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--recipe", type=Path, required=True)
    seal.add_argument("--stage-receipt", action="append", default=[])
    seal.add_argument("--cache-receipt", action="append", default=[])
    seal.add_argument("--event-parent-receipt", action="append", default=[])
    seal.add_argument("--normalizer", action="append", default=[])
    seal.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-head")
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--fit-directory", type=Path, required=True)
    export.add_argument("--model-seal", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--inputs", type=Path, required=True)
    package.add_argument("--predictions", type=Path, required=True)
    package.add_argument("--prediction-receipt", type=Path, required=True)
    package.add_argument("--model-seal", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seal":
        result = seal_final_chain(
            recipe_path=args.recipe,
            stage_receipts=_pairs(args.stage_receipt),
            cache_receipts=_pairs(args.cache_receipt),
            event_parent_receipts=_pairs(args.event_parent_receipt),
            normalizers=_pairs(args.normalizer),
            output=args.output,
        )
    elif args.command == "export-head":
        result = export_phase_only_head(
            recipe_path=args.recipe,
            fit_directory=args.fit_directory,
            model_seal_path=args.model_seal,
            output=args.output,
        )
    else:
        result = package_final_predictions(
            inputs_path=args.inputs,
            predictions_path=args.predictions,
            prediction_receipt_path=args.prediction_receipt,
            model_seal_path=args.model_seal,
            output=args.output,
            receipt_output=args.receipt_output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "export_phase_only_head",
    "main",
    "package_final_predictions",
    "seal_final_chain",
]
