"""Validated training unit for the inner PAIR/CTX stages of one OOF rotation.

This module deliberately stops before feature generation, stitching, final-head
training, deployment refits, evaluation, or queue activation.  It admits only
rotation-local PAIR133 caches on the producer role and PHASE17 histories on the
disjoint refiner role.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from operational.rgb_port.accounting import read_json_shared

from .features import validate_cache_receipt
from .recipe import FinalRefitRecipe, StageRecipe, canonical_sha256, sha256
from .training import CachedTrainingScope, fit_cached_stage

if TYPE_CHECKING:
    import pandas as pd

SCHEMA = "final_refit_fold_training_unit_v1"
ENDPOINT_SCHEMA = "final_refit_fold_endpoint_admission_v1"
ROLE_TO_CACHE_ROLE = {"producer": "producer_pair", "refiner": "refiner"}
TECHNICAL_LIFETIME_UPDATE_CAP = 10
TECHNICAL_CALL_UPDATE_CAP = 5


@dataclass(frozen=True)
class FoldStage:
    """One trainable inner stage with its exact fold role and endpoint."""

    stage_id: str
    kind: str
    role: str
    population: int
    groups: int
    updates: int
    batch_size: int
    weight_decay: float
    depends_on: tuple[str, ...]
    cache_schema: str

    def recipe(self) -> StageRecipe:
        """Convert to the model/loss contract consumed by the common engine."""
        return StageRecipe(
            self.stage_id,
            self.kind,
            self.updates,
            self.batch_size,
            self.weight_decay,
            self.depends_on,
            True,
            self.cache_schema,
        )


@dataclass(frozen=True)
class FoldTrainingUnit:
    """Immutable identity of one rotation's four inner head fits."""

    schema: str
    status: str
    rotation: int
    fold_contract_path: str
    fold_contract_sha256: str
    rotation_manifest_path: str
    rotation_manifest_sha256: str
    roles: dict[str, dict[str, Any]]
    stages: tuple[FoldStage, ...]
    executed_source_sha256: dict[str, str]
    exclusions: tuple[str, ...]
    unit_sha256: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("unit_sha256")
        return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _role_frame(path: Path, *, rotation: int, role: str) -> pd.DataFrame:
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {
        "rotation",
        "rotation_role",
        "source_ordinal",
        "sample_token",
        "group_id",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{role} manifest lacks columns: {sorted(required - set(frame.columns))}")
    if (
        set(frame["rotation"].astype(int).tolist()) != {rotation}
        or set(frame["rotation_role"].astype(str).tolist()) != {role}
        or frame["sample_token"].astype(str).duplicated().any()
        or frame["source_ordinal"].astype(int).duplicated().any()
    ):
        raise ValueError(f"Rotation {rotation} {role} rows differ from their declared role")
    return frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)


def _token_sha(tokens: list[str]) -> str:
    return canonical_sha256(tokens)


def build_fold_training_unit(
    fold_contract_path: Path,
    *,
    rotation: int,
    executed_sources: Mapping[str, Path],
) -> FoldTrainingUnit:
    """Bind one verified 24/8/8 rotation without reading training targets."""
    contract_path = fold_contract_path.resolve(strict=True)
    contract = _read(contract_path)
    if (
        contract.get("schema") != "final_refit_oof_fold_contract_v1"
        or contract.get("status") != "PASS"
        or contract.get("training_started") is not False
        or contract.get("optimizer_updates") != 0
        or contract.get("generation_coverage") != "EACH_TRAIN40_TOKEN_EXACTLY_ONCE"
    ):
        raise ValueError("FOLD_CONTRACT is not an unused verified OOF partition")
    rotations = contract.get("rotations")
    if (
        contract.get("population") != {"queries": 88_744, "groups": 40}
        or not isinstance(rotations, list)
        or len(rotations) != 5
        or rotation not in range(len(rotations))
    ):
        raise ValueError("Unknown OOF rotation")
    manifest_ref = rotations[rotation]
    manifest_path = (contract_path.parent / str(manifest_ref["path"])).resolve(strict=True)
    if sha256(manifest_path) != manifest_ref.get("sha256"):
        raise ValueError("Rotation manifest changed after FOLD_CONTRACT")
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != "final_refit_rotation_roles_v1"
        or manifest.get("status") != "VERIFIED_REFERENCE_MANIFESTS"
        or manifest.get("rotation") != rotation
        or manifest.get("generation_ancestry_excludes_group") is not True
    ):
        raise ValueError("Rotation role manifest is not admissible")

    roles: dict[str, dict[str, Any]] = {}
    group_sets: dict[str, set[str]] = {}
    token_sets: dict[str, set[str]] = {}
    ordinal_sets: dict[str, set[int]] = {}
    for role in ("producer", "refiner", "generation"):
        reference = manifest["roles"][role]
        path = (manifest_path.parent / str(reference["path"])).resolve(strict=True)
        if sha256(path) != reference.get("sha256"):
            raise ValueError(f"Rotation {rotation} {role} rows changed")
        frame = _role_frame(path, rotation=rotation, role=role)
        groups = {str(value) for value in frame["group_id"].tolist()}
        tokens = [str(value) for value in frame["sample_token"].tolist()]
        ordinals = [str(value) for value in frame["source_ordinal"].astype(int).tolist()]
        if len(frame) != int(reference["queries"]) or len(groups) != int(reference["groups"]):
            raise ValueError(f"Rotation {rotation} {role} population changed")
        roles[role] = {
            "path": str(path),
            "sha256": sha256(path),
            "population": len(frame),
            "groups": len(groups),
            "group_ids": sorted(groups),
            "sample_token_sha256": _token_sha(tokens),
            "sample_token_set_sha256": _token_sha(sorted(tokens)),
            "ordered_native_ordinal_sha256": _token_sha(ordinals),
        }
        group_sets[role] = groups
        token_sets[role] = set(tokens)
        ordinal_sets[role] = {int(value) for value in ordinals}
    if any(
        group_sets[left] & group_sets[right]
        or token_sets[left] & token_sets[right]
        or ordinal_sets[left] & ordinal_sets[right]
        for left, right in (
            ("producer", "refiner"),
            ("producer", "generation"),
            ("refiner", "generation"),
        )
    ):
        raise ValueError("Rotation roles overlap by group or sample token")
    if (len(group_sets["producer"]), len(group_sets["refiner"]), len(group_sets["generation"])) != (
        24,
        8,
        8,
    ):
        raise ValueError("Rotation is not the required 24/8/8 group partition")
    if sum(role["population"] for role in roles.values()) != 88_744:
        raise ValueError("Rotation roles do not cover the full TRAIN40 population")
    if set().union(*ordinal_sets.values()) != set(range(88_744)):
        raise ValueError("Rotation roles do not cover native TRAIN40 ordinals exactly")

    producer_n = roles["producer"]["population"]
    refiner_n = roles["refiner"]["population"]
    pair_updates = math.ceil(6_840 * producer_n / 53_890)
    context_updates = math.ceil(2_500 * refiner_n / 18_422)
    stages = (
        FoldStage(
            "PAIR_E",
            "pair_head",
            "producer_pair",
            producer_n,
            24,
            pair_updates,
            32,
            1e-4,
            ("E_A5",),
            "final_refit_pair_cache_v1",
        ),
        FoldStage(
            "PAIR_R",
            "pair_head",
            "producer_pair",
            producer_n,
            24,
            pair_updates,
            32,
            1e-4,
            ("R_A5",),
            "final_refit_pair_cache_v1",
        ),
        FoldStage(
            "E_CTX",
            "temporal_head",
            "refiner",
            refiner_n,
            8,
            context_updates,
            128,
            1e-3,
            ("E_A5", "E_C2F", "PAIR_E"),
            "final_refit_context_cache_v1",
        ),
        FoldStage(
            "R_CTX",
            "temporal_head",
            "refiner",
            refiner_n,
            8,
            context_updates,
            128,
            1e-3,
            ("R_A5", "R_C2F", "PAIR_R"),
            "final_refit_context_cache_v1",
        ),
    )
    if not executed_sources:
        raise ValueError("Fold unit must bind all executed sources")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_QUEUED",
        "rotation": rotation,
        "fold_contract_path": str(contract_path),
        "fold_contract_sha256": sha256(contract_path),
        "rotation_manifest_path": str(manifest_path),
        "rotation_manifest_sha256": sha256(manifest_path),
        "roles": roles,
        "stages": [asdict(stage) for stage in stages],
        "executed_source_sha256": {
            name: sha256(path.resolve(strict=True))
            for name, path in sorted(executed_sources.items())
        },
        "exclusions": [
            "scientific fold fitting until transitive OOF closure is admitted",
            "producer training",
            "generation inference",
            "cross-rotation stitching",
            "final global normalizer",
            "final context refit",
            "PHASE_ONLY training",
            "evaluation",
            "submission",
        ],
    }
    return FoldTrainingUnit(
        **{**payload, "stages": stages, "exclusions": tuple(payload["exclusions"])},
        unit_sha256=canonical_sha256(payload),
    )


def write_fold_training_unit(path: Path, unit: FoldTrainingUnit) -> None:
    """Create a unit atomically and refuse identity-changing replacement."""
    encoded = json.dumps(asdict(unit), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Refusing to replace a changed fold training unit")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_fold_training_unit(path: Path) -> FoldTrainingUnit:
    """Reload a unit and verify its identity, sources, folds, and exclusions."""
    value = _read(path.resolve(strict=True))
    stages = tuple(
        FoldStage(**{**item, "depends_on": tuple(item["depends_on"])})
        for item in value.pop("stages")
    )
    exclusions = tuple(value.pop("exclusions"))
    unit = FoldTrainingUnit(stages=stages, exclusions=exclusions, **value)
    if (
        unit.schema != SCHEMA
        or unit.status != "PREPARED_NOT_QUEUED"
        or canonical_sha256(unit.payload()) != unit.unit_sha256
    ):
        raise ValueError("Fold training unit identity changed")
    for name, digest in unit.executed_source_sha256.items():
        source = Path(name)
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[2] / source
        if sha256(source.resolve(strict=True)) != digest:
            raise ValueError(f"Fold executed source changed: {name}")
    if sha256(Path(unit.fold_contract_path)) != unit.fold_contract_sha256:
        raise ValueError("FOLD_CONTRACT changed after unit preparation")
    if sha256(Path(unit.rotation_manifest_path)) != unit.rotation_manifest_sha256:
        raise ValueError("Rotation manifest changed after unit preparation")
    return unit


def _stage(unit: FoldTrainingUnit, stage_id: str) -> FoldStage:
    matches = [stage for stage in unit.stages if stage.stage_id == stage_id]
    if len(matches) != 1 or stage_id not in {"PAIR_E", "PAIR_R", "E_CTX", "R_CTX"}:
        raise ValueError("Fold unit contains no unique supported inner stage")
    return matches[0]


def _validate_cache(unit: FoldTrainingUnit, stage: FoldStage, path: Path) -> None:
    receipt = validate_cache_receipt(path, stage.cache_schema, expected_role=stage.role)
    role_name = "producer" if stage.role == "producer_pair" else "refiner"
    identity = unit.roles[role_name]
    if (
        receipt.get("rows") != stage.population
        or receipt.get("groups") != stage.groups
        or sorted(str(item) for item in receipt.get("group_ids", [])) != identity["group_ids"]
    ):
        raise ValueError("Fold cache population or group role differs")
    with np.load(path, allow_pickle=False) as archive:
        tokens = np.asarray(archive["sample_token"]).astype(str).tolist()
        roles = set(np.asarray(archive["role"]).astype(str).tolist())
    if roles != {stage.role} or _token_sha(tokens) != identity["sample_token_sha256"]:
        raise ValueError("Fold cache tokens or explicit role differ from its manifest")
    if stage.kind == "temporal_head" and (
        receipt.get("normalizer_fit_role") != "refiner"
        or sorted(str(item) for item in receipt.get("normalizer_group_ids", []))
        != unit.roles["refiner"]["group_ids"]
    ):
        raise ValueError("Context PHASE17 normalizer was not fit only on refiner groups")


def _validate_ancestors(
    unit: FoldTrainingUnit,
    stage: FoldStage,
    parent_receipts: Mapping[str, Path],
    ancestor_admissions: Mapping[str, Path],
) -> dict[str, str]:
    if set(parent_receipts) != set(stage.depends_on) or set(ancestor_admissions) != set(
        stage.depends_on
    ):
        raise ValueError("Fold parent receipts/admissions differ from stage dependencies")
    expected_groups = unit.roles["producer"]["group_ids"]
    checkpoint_hashes: dict[str, str] = {}
    for name in stage.depends_on:
        receipt_path = parent_receipts[name].resolve(strict=True)
        receipt = read_json_shared(receipt_path)
        admission_path = ancestor_admissions[name].resolve(strict=True)
        admission = _read(admission_path)
        payload = dict(admission)
        claimed = payload.pop("admission_payload_sha256", None)
        checkpoint_path = Path(str(receipt.get("checkpoint_path", ""))).resolve(strict=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_identity = checkpoint.get("identity") if isinstance(checkpoint, dict) else None
        identity_payload = (
            {key: value for key, value in checkpoint_identity.items() if key != "identity_sha256"}
            if isinstance(checkpoint_identity, dict)
            else None
        )
        expected_identity_schema = (
            "final_refit_fold_producer_identity_v1"
            if name in {"E_A5", "E_C2F", "R_A5", "R_C2F"}
            else "final_refit_cached_stage_identity_v1"
        )
        parent_maps = (
            admission.get("parent_checkpoint_sha256"),
            admission.get("parent_receipt_sha256"),
            admission.get("parent_admission_sha256"),
        )
        if (
            admission.get("schema") != ENDPOINT_SCHEMA
            or admission.get("status") != "PASS"
            or admission.get("technical") is not False
            or admission.get("scientific_endpoint") is not True
            or admission.get("rotation") != unit.rotation
            or admission.get("trained_role") != "producer_pair"
            or admission.get("fit_id") != name
            or admission.get("fold_contract_sha256") != unit.fold_contract_sha256
            or admission.get("rotation_manifest_sha256") != unit.rotation_manifest_sha256
            or admission.get("train_rows_sha256") != unit.roles["producer"]["sha256"]
            or admission.get("ordered_sample_token_sha256")
            != unit.roles["producer"]["sample_token_sha256"]
            or admission.get("sample_token_set_sha256")
            != unit.roles["producer"]["sample_token_set_sha256"]
            or admission.get("ordered_native_ordinal_sha256")
            != unit.roles["producer"]["ordered_native_ordinal_sha256"]
            or admission.get("checkpoint_path") != str(checkpoint_path)
            or admission.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
            or admission.get("train_group_ids") != expected_groups
            or admission.get("checkpoint_receipt_path") != str(receipt_path)
            or admission.get("checkpoint_receipt_sha256") != sha256(receipt_path)
            or admission.get("training_cache_role") != "producer_pair"
            or admission.get("generation_group_ids") != unit.roles["generation"]["group_ids"]
            or admission.get("generation_token_set_sha256")
            != unit.roles["generation"]["sample_token_set_sha256"]
            or admission.get("generation_rows_excluded_from_all_ancestors") is not True
            or not all(isinstance(item, dict) for item in parent_maps)
            or claimed != canonical_sha256(payload)
            or receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("scientific_endpoint") is not True
            or receipt.get("fit_id") != name
            or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
            or not isinstance(checkpoint_identity, dict)
            or checkpoint_identity.get("schema") != expected_identity_schema
            or checkpoint_identity.get("fit_id") != name
            or checkpoint_identity.get("technical") is not False
            or checkpoint_identity.get("identity_sha256") != receipt.get("identity_sha256")
            or checkpoint.get("identity_sha256") != receipt.get("identity_sha256")
            or canonical_sha256(identity_payload) != receipt.get("identity_sha256")
            or admission.get("checkpoint_identity_sha256") != receipt.get("identity_sha256")
            or checkpoint_identity.get("training_cache_sha256")
            != admission.get("training_cache_sha256")
            or checkpoint_identity.get("training_sample_token_sha256")
            != admission.get("ordered_sample_token_sha256")
        ):
            raise ValueError(f"Fold ancestor {name} lacks exact producer-role ancestry")
        checkpoint_hashes[name] = str(receipt["checkpoint_sha256"])
    return checkpoint_hashes


def _validate_technical_receipt(receipt: Mapping[str, Any]) -> None:
    """Reject any technical state that reaches or claims a scientific endpoint."""
    if (
        receipt.get("status") == "COMPLETE"
        or receipt.get("scientific_endpoint") is not False
        or int(receipt.get("completed_updates", -1)) > TECHNICAL_LIFETIME_UPDATE_CAP
    ):
        raise RuntimeError("Technical fold training attempted to create a scientific endpoint")


def fit_fold_stage(
    *,
    unit_path: Path,
    base_recipe: FinalRefitRecipe,
    base_recipe_path: Path,
    stage_id: str,
    cache_path: Path,
    parent_receipts: Mapping[str, Path],
    ancestor_admissions: Mapping[str, Path],
    run_root: Path,
    device_name: str = "cuda",
    max_updates_this_call: int | None = None,
    resident_cache: bool = True,
    technical: bool = False,
) -> dict[str, Any]:
    """Validate and delegate one bounded fold fit to the common durable engine."""
    unit = load_fold_training_unit(unit_path)
    stage = _stage(unit, stage_id)
    if not technical:
        raise RuntimeError(
            "Scientific fold fitting is fail-closed: producer-to-pair-to-context transitive "
            "ancestry and deployment-refit closure are not yet admitted"
        )
    if (
        max_updates_this_call is None
        or max_updates_this_call < 1
        or max_updates_this_call > TECHNICAL_CALL_UPDATE_CAP
    ):
        raise ValueError("Fold technical call requires between one and five updates")
    if stage.updates <= TECHNICAL_LIFETIME_UPDATE_CAP:
        raise ValueError("Fold scientific endpoint is too small for isolated technical probing")
    technical_directory = run_root / f"rotation_{unit.rotation}" / "technical" / stage.stage_id
    prior_receipt_path = technical_directory / "CHECKPOINT_RECEIPT.json"
    prior_updates = 0
    if prior_receipt_path.is_file():
        prior_receipt = read_json_shared(prior_receipt_path)
        prior_updates = int(prior_receipt.get("completed_updates", -1))
        if (
            prior_receipt.get("status") == "COMPLETE"
            or prior_receipt.get("scientific_endpoint") is not False
            or prior_updates < 0
        ):
            raise ValueError("Existing fold technical receipt claims an endpoint")
    if prior_updates + max_updates_this_call > TECHNICAL_LIFETIME_UPDATE_CAP:
        raise ValueError("Fold technical lifetime update cap would be exceeded")
    cache = cache_path.resolve(strict=True)
    _validate_cache(unit, stage, cache)
    ancestor_checkpoints = _validate_ancestors(unit, stage, parent_receipts, ancestor_admissions)
    scope = CachedTrainingScope(
        population=stage.population,
        groups=stage.groups,
        role=stage.role,
        experiment_id=f"FINAL_REFIT_OOF_ROTATION_{unit.rotation}",
        split_version=f"TRAIN40_NESTED_ROTATION_{unit.rotation}_{stage.role.upper()}",
        scope_sha256=unit.unit_sha256,
        external_parent_sha256={
            name: digest
            for name, digest in ancestor_checkpoints.items()
            if name in {"E_A5", "E_C2F"}
        },
    )
    receipt = fit_cached_stage(
        recipe=base_recipe,
        recipe_path=base_recipe_path,
        stage=stage.recipe(),
        cache_path=cache,
        parent_receipts=parent_receipts,
        run_root=run_root / f"rotation_{unit.rotation}",
        device_name=device_name,
        max_updates_this_call=max_updates_this_call,
        resident_cache=resident_cache,
        technical=technical,
        cache_scope=scope,
    )
    _validate_technical_receipt(receipt)
    return receipt


__all__ = [
    "ENDPOINT_SCHEMA",
    "TECHNICAL_CALL_UPDATE_CAP",
    "TECHNICAL_LIFETIME_UPDATE_CAP",
    "FoldStage",
    "FoldTrainingUnit",
    "build_fold_training_unit",
    "fit_fold_stage",
    "load_fold_training_unit",
    "write_fold_training_unit",
]
