"""Fail-closed lineage contracts for grouped OOF feature generation and stitching.

The contracts in this module describe temporary models used to create training
features.  They deliberately keep that ancestry separate from the checkpoints
that a final full-TRAIN deployment will execute.  This module performs no
training, inference, target access, or queue activation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ENDPOINT_SCHEMA = "final_refit_oof_endpoint_lineage_v1"
STITCH_SCHEMA = "final_refit_oof_lineage_stitch_v1"
TRANSITION_SCHEMA = "final_refit_oof_deployment_transition_v1"
CONTRACT_SCHEMA = "final_refit_oof_fold_contract_v1"
MANIFEST_SCHEMA = "final_refit_rotation_roles_v1"

STAGE_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "E_A5": ("producer", ()),
    "E_C2F": ("producer", ()),
    "R_A5": ("producer", ()),
    "R_C2F": ("producer", ()),
    "PAIR_E": ("producer", ("E_A5",)),
    "PAIR_R": ("producer", ("R_A5",)),
    "E_CTX": ("refiner", ("E_A5", "E_C2F", "PAIR_E")),
    "R_CTX": ("refiner", ("R_A5", "R_C2F", "PAIR_R")),
}
INTERFACES = {
    "PHASE17_RAW_EVENT": (17, ("E_A5", "E_C2F", "PAIR_E")),
    "PHASE17_RAW_RGB": (17, ("R_A5", "R_C2F", "PAIR_R")),
    "PHASE20_CONTEXT": (20, ("E_CTX", "R_CTX")),
}
FORBIDDEN_REPRESENTATIONS = ("latent", "embedding", "token128", "dense_token")


def file_sha256(path: Path) -> str:
    """Hash a required immutable file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON without accepting NaN."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def ordered_identity_sha256(values: Sequence[str]) -> str:
    """Hash an ordered identity vector without delimiter ambiguity."""
    return canonical_sha256([str(value) for value in values])


def set_identity_sha256(values: Sequence[str]) -> str:
    """Hash a unique identity set in stable order."""
    strings = [str(value) for value in values]
    if len(strings) != len(set(strings)):
        raise ValueError("Identity set contains duplicates")
    return canonical_sha256(sorted(strings))


def _digest(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _artifact_reference(path: Path, claimed_sha256: str, *, name: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    claimed = _digest(claimed_sha256, name=name)
    actual = file_sha256(resolved)
    if actual != claimed:
        raise ValueError(f"{name} bytes differ from the declared SHA-256")
    return {"path": str(resolved), "sha256": actual}


def _verify_artifact_reference(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} artifact reference differs")
    path = Path(str(value["path"])).resolve(strict=True)
    digest = _digest(str(value["sha256"]), name=name)
    if file_sha256(path) != digest:
        raise ValueError(f"{name} artifact bytes changed")
    return {"path": str(path), "sha256": digest}


@dataclass(frozen=True)
class RoleIdentity:
    """Exact row identity of one role in one rotation."""

    rotation: int
    role: str
    manifest_path: str
    manifest_sha256: str
    rows_path: str
    rows_sha256: str
    population: int
    group_ids: tuple[str, ...]
    ordered_sample_token_sha256: str
    sample_token_set_sha256: str
    ordered_native_ordinal_sha256: str


@dataclass(frozen=True)
class RotationIdentity:
    """The three disjoint 24/8/8 identities for one rotation."""

    fold_contract_path: str
    fold_contract_sha256: str
    rotation: int
    roles: dict[str, RoleIdentity]


def load_rotation_identity(fold_contract_path: Path, rotation: int) -> RotationIdentity:
    """Reload and verify exact role rows referenced by ``FOLD_CONTRACT``."""
    contract_path = fold_contract_path.resolve(strict=True)
    contract = _read(contract_path)
    rotations = contract.get("rotations")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("training_started") is not False
        or contract.get("optimizer_updates") != 0
        or not isinstance(rotations, list)
        or rotation not in range(len(rotations))
    ):
        raise ValueError("Fold contract is not an unused verified OOF partition")
    reference = rotations[rotation]
    manifest_path = (contract_path.parent / str(reference["path"])).resolve(strict=True)
    if file_sha256(manifest_path) != reference.get("sha256"):
        raise ValueError("Rotation manifest changed after fold contract")
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "VERIFIED_REFERENCE_MANIFESTS"
        or manifest.get("rotation") != rotation
        or manifest.get("generation_ancestry_excludes_group") is not True
    ):
        raise ValueError("Rotation manifest is not an admitted grouped partition")

    identities: dict[str, RoleIdentity] = {}
    token_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}
    for role in ("producer", "refiner", "generation"):
        role_reference = manifest["roles"][role]
        rows_path = (manifest_path.parent / str(role_reference["path"])).resolve(strict=True)
        rows_sha = file_sha256(rows_path)
        if rows_sha != role_reference.get("sha256"):
            raise ValueError(f"Rotation {rotation} {role} rows changed")
        frame = pd.read_parquet(
            rows_path,
            columns=[
                "rotation",
                "rotation_role",
                "source_ordinal",
                "sample_token",
                "group_id",
            ],
        )
        tokens = frame["sample_token"].astype(str).tolist()
        groups = frame["group_id"].astype(str).tolist()
        ordinals = frame["source_ordinal"].astype(int).tolist()
        if (
            len(frame) != int(role_reference["queries"])
            or frame["sample_token"].astype(str).duplicated().any()
            or frame["source_ordinal"].astype(int).duplicated().any()
            or set(frame["rotation"].astype(int)) != {rotation}
            or set(frame["rotation_role"].astype(str)) != {role}
            or len(set(groups)) != int(role_reference["groups"])
        ):
            raise ValueError(f"Rotation {rotation} {role} identity differs")
        identities[role] = RoleIdentity(
            rotation=rotation,
            role=role,
            manifest_path=str(manifest_path),
            manifest_sha256=file_sha256(manifest_path),
            rows_path=str(rows_path),
            rows_sha256=rows_sha,
            population=len(frame),
            group_ids=tuple(sorted(set(groups))),
            ordered_sample_token_sha256=ordered_identity_sha256(tokens),
            sample_token_set_sha256=set_identity_sha256(tokens),
            ordered_native_ordinal_sha256=ordered_identity_sha256(
                [str(value) for value in ordinals]
            ),
        )
        token_sets[role] = set(tokens)
        group_sets[role] = set(groups)
    role_pairs = (
        ("producer", "refiner"),
        ("producer", "generation"),
        ("refiner", "generation"),
    )
    for left, right in role_pairs:
        if token_sets[left] & token_sets[right] or group_sets[left] & group_sets[right]:
            raise ValueError("Rotation roles overlap by row or group")
    if tuple(len(group_sets[role]) for role in ("producer", "refiner", "generation")) != (
        24,
        8,
        8,
    ):
        raise ValueError("Rotation is not the required 24/8/8 partition")
    return RotationIdentity(
        fold_contract_path=str(contract_path),
        fold_contract_sha256=file_sha256(contract_path),
        rotation=rotation,
        roles=identities,
    )


def _lineage_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("lineage_sha256", None)
    return payload


def _json_equivalent(left: object, right: object) -> bool:
    """Compare values after the tuple-to-list conversion performed by JSON."""
    return canonical_sha256(left) == canonical_sha256(right)


def _fold_universe(
    fold_contract_path: Path,
) -> tuple[int, int, set[str], set[str], set[int]]:
    """Recover the exact row/group universe and verify five-fold generation coverage."""
    identities = [load_rotation_identity(fold_contract_path, rotation) for rotation in range(5)]
    reference_tokens: set[str] = set()
    reference_groups: set[str] = set()
    reference_rows = 0
    reference_ordinals: set[int] = set()
    for role in ("producer", "refiner", "generation"):
        rows = pd.read_parquet(
            identities[0].roles[role].rows_path,
            columns=["sample_token", "source_ordinal"],
        )
        tokens = set(rows["sample_token"].astype(str))
        ordinals = set(rows["source_ordinal"].astype(int))
        if reference_tokens & tokens or reference_ordinals & ordinals:
            raise ValueError("Fold reference roles overlap by sample token or source ordinal")
        reference_tokens.update(tokens)
        reference_ordinals.update(ordinals)
        reference_groups.update(identities[0].roles[role].group_ids)
        reference_rows += identities[0].roles[role].population

    generated_tokens: set[str] = set()
    generated_groups: set[str] = set()
    generated_rows = 0
    generated_ordinals: set[int] = set()
    for identity in identities:
        generation = identity.roles["generation"]
        rows = pd.read_parquet(
            generation.rows_path,
            columns=["sample_token", "source_ordinal"],
        )
        tokens = set(rows["sample_token"].astype(str))
        ordinals = set(rows["source_ordinal"].astype(int))
        if (
            generated_tokens & tokens
            or generated_groups & set(generation.group_ids)
            or generated_ordinals & ordinals
        ):
            raise ValueError("OOF generation folds overlap by token, group or source ordinal")
        generated_tokens.update(tokens)
        generated_groups.update(generation.group_ids)
        generated_ordinals.update(ordinals)
        generated_rows += generation.population

    if (
        generated_rows != reference_rows
        or generated_tokens != reference_tokens
        or generated_groups != reference_groups
        or generated_ordinals != reference_ordinals
        or reference_ordinals != set(range(reference_rows))
    ):
        raise ValueError("OOF generation folds do not cover the fold universe exactly once")

    contract = _read(fold_contract_path)
    declared = contract.get("population")
    if declared is not None:
        if not isinstance(declared, dict) or (
            int(declared.get("queries", -1)) != reference_rows
            or int(declared.get("groups", -1)) != len(reference_groups)
        ):
            raise ValueError("Fold contract population differs from its exact row universe")
    return (
        reference_rows,
        len(reference_groups),
        reference_tokens,
        reference_groups,
        reference_ordinals,
    )


def verify_endpoint_lineage(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Verify a temporary endpoint and every declared training ancestor."""
    payload = _read(value.resolve(strict=True)) if isinstance(value, Path) else dict(value)
    claimed = payload.get("lineage_sha256")
    if (
        payload.get("schema") != ENDPOINT_SCHEMA
        or payload.get("status")
        != "VERIFIED_STRUCTURAL_OOF_ENDPOINT_NOT_SCIENTIFICALLY_ADMITTED"
        or payload.get("artifact_files_rehashed") is not True
        or payload.get("targets_used_for_partitioning") is not False
        or payload.get("blind_validation_claim") is not False
        or payload.get("scientific_launch_authorized") is not False
        or claimed != canonical_sha256(_lineage_payload(payload))
    ):
        raise ValueError("OOF endpoint lineage identity differs")
    stage_id = str(payload.get("stage_id"))
    if stage_id not in STAGE_SPEC:
        raise ValueError("Unknown OOF stage")
    role_name, parent_names = STAGE_SPEC[stage_id]
    identity = load_rotation_identity(
        Path(str(payload["fold_contract_path"])), int(payload["rotation"])
    )
    role = identity.roles[role_name]
    generation = identity.roles["generation"]
    own = payload.get("own_training_rows")
    if not isinstance(own, dict) or not _json_equivalent(own, asdict(role)):
        raise ValueError("OOF endpoint training rows differ from its exact fold role")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "checkpoint",
        "checkpoint_receipt",
        "training_cache",
    }:
        raise ValueError("OOF endpoint artifact bindings differ")
    checkpoint = _verify_artifact_reference(artifacts["checkpoint"], name="checkpoint")
    checkpoint_receipt = _verify_artifact_reference(
        artifacts["checkpoint_receipt"], name="checkpoint_receipt"
    )
    training_cache = _verify_artifact_reference(
        artifacts["training_cache"], name="training_cache"
    )
    if (
        payload.get("checkpoint_sha256") != checkpoint["sha256"]
        or payload.get("checkpoint_receipt_sha256") != checkpoint_receipt["sha256"]
    ):
        raise ValueError("OOF endpoint artifact hashes differ from their bindings")
    training = payload.get("training_feature_ancestry")
    deployment = payload.get("deployment_dependencies")
    if not isinstance(training, dict) or deployment != {}:
        raise ValueError("Temporary OOF endpoint mixes training ancestry with deployment parents")
    if set(training.get("direct_parents", {})) != set(parent_names):
        raise ValueError("OOF endpoint direct parents differ from its stage graph")
    if set(training.get("direct_parent_lineages", {})) != set(parent_names):
        raise ValueError("OOF endpoint parent lineage set differs from its stage graph")
    for name in parent_names:
        reference = training["direct_parent_lineages"][name]
        parent_path = Path(str(reference["path"])).resolve(strict=True)
        if file_sha256(parent_path) != reference.get("file_sha256"):
            raise ValueError("OOF parent lineage file changed")
        parent = verify_endpoint_lineage(parent_path)
        if (
            parent.get("stage_id") != name
            or parent.get("rotation") != identity.rotation
            or parent.get("fold_contract_sha256") != identity.fold_contract_sha256
            or parent.get("checkpoint_sha256") != training["direct_parents"][name]
            or parent.get("lineage_sha256") != reference.get("lineage_sha256")
        ):
            raise ValueError("OOF parent lineage does not match the declared direct parent")
    ancestor_roles = ["producer"] if role_name == "producer" else ["producer", "refiner"]
    ancestor_groups = sorted(
        set().union(*(set(identity.roles[name].group_ids) for name in ancestor_roles))
    )
    ancestor_token_hashes = {
        name: identity.roles[name].sample_token_set_sha256 for name in ancestor_roles
    }
    if (
        training.get("ancestor_train_group_ids") != ancestor_groups
        or training.get("ancestor_role_token_set_sha256") != ancestor_token_hashes
        or set(ancestor_groups) & set(generation.group_ids)
        or training.get("generation_group_ids") != list(generation.group_ids)
        or training.get("generation_token_set_sha256") != generation.sample_token_set_sha256
        or training.get("generation_rows_excluded_from_all_ancestors") is not True
    ):
        raise ValueError("Generation rows are not proven absent from complete training ancestry")
    if training.get("training_cache_sha256") != training_cache["sha256"]:
        raise ValueError("OOF endpoint training cache differs from its binding")
    return payload


def build_endpoint_lineage(
    *,
    fold_contract_path: Path,
    rotation: int,
    stage_id: str,
    checkpoint_sha256: str,
    checkpoint_receipt_sha256: str,
    training_cache_sha256: str,
    checkpoint_path: Path,
    checkpoint_receipt_path: Path,
    training_cache_path: Path,
    parent_lineage_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a signed endpoint lineage from exact fold roles and parent files."""
    if stage_id not in STAGE_SPEC:
        raise ValueError("Unknown OOF stage")
    role_name, parent_names = STAGE_SPEC[stage_id]
    if set(parent_lineage_paths) != set(parent_names):
        raise ValueError("Parent lineage paths differ from stage graph")
    identity = load_rotation_identity(fold_contract_path, rotation)
    artifacts = {
        "checkpoint": _artifact_reference(
            checkpoint_path, checkpoint_sha256, name="checkpoint_sha256"
        ),
        "checkpoint_receipt": _artifact_reference(
            checkpoint_receipt_path,
            checkpoint_receipt_sha256,
            name="checkpoint_receipt_sha256",
        ),
        "training_cache": _artifact_reference(
            training_cache_path, training_cache_sha256, name="training_cache_sha256"
        ),
    }
    direct_parents: dict[str, str] = {}
    references: dict[str, dict[str, str]] = {}
    for name in parent_names:
        path = parent_lineage_paths[name].resolve(strict=True)
        parent = verify_endpoint_lineage(path)
        direct_parents[name] = str(parent["checkpoint_sha256"])
        references[name] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "lineage_sha256": str(parent["lineage_sha256"]),
        }
    ancestor_roles = ["producer"] if role_name == "producer" else ["producer", "refiner"]
    training = {
        "training_cache_sha256": _digest(training_cache_sha256, name="training_cache_sha256"),
        "direct_parents": direct_parents,
        "direct_parent_lineages": references,
        "ancestor_train_group_ids": sorted(
            set().union(*(set(identity.roles[name].group_ids) for name in ancestor_roles))
        ),
        "ancestor_role_token_set_sha256": {
            name: identity.roles[name].sample_token_set_sha256 for name in ancestor_roles
        },
        "generation_group_ids": list(identity.roles["generation"].group_ids),
        "generation_token_set_sha256": identity.roles["generation"].sample_token_set_sha256,
        "generation_rows_excluded_from_all_ancestors": True,
    }
    payload: dict[str, Any] = {
        "schema": ENDPOINT_SCHEMA,
        "status": "VERIFIED_STRUCTURAL_OOF_ENDPOINT_NOT_SCIENTIFICALLY_ADMITTED",
        "rotation": rotation,
        "stage_id": stage_id,
        "fold_contract_path": identity.fold_contract_path,
        "fold_contract_sha256": identity.fold_contract_sha256,
        "own_training_rows": asdict(identity.roles[role_name]),
        "checkpoint_sha256": _digest(checkpoint_sha256, name="checkpoint_sha256"),
        "checkpoint_receipt_sha256": _digest(
            checkpoint_receipt_sha256, name="checkpoint_receipt_sha256"
        ),
        "artifacts": artifacts,
        "training_feature_ancestry": training,
        "deployment_dependencies": {},
        "artifact_files_rehashed": True,
        "targets_used_for_partitioning": False,
        "blind_validation_claim": False,
        "scientific_launch_authorized": False,
    }
    result = {**payload, "lineage_sha256": canonical_sha256(payload)}
    return verify_endpoint_lineage(result)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace changed lineage receipt: {path}")
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


def write_endpoint_lineage(path: Path, value: Mapping[str, Any]) -> None:
    """Persist one verified immutable endpoint lineage."""
    _atomic_json(path, verify_endpoint_lineage(value))


def _contains_forbidden_representation(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(term in str(key).lower() for term in FORBIDDEN_REPRESENTATIONS)
            or _contains_forbidden_representation(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_representation(item) for item in value)
    return False


def verify_stitch_receipt(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Verify exact five-fold coverage at a physically commensurate interface."""
    receipt = _read(value.resolve(strict=True)) if isinstance(value, Path) else dict(value)
    claimed = receipt.get("stitch_sha256")
    unhashed = dict(receipt)
    unhashed.pop("stitch_sha256", None)
    interface = str(receipt.get("interface"))
    if (
        receipt.get("schema") != STITCH_SCHEMA
        or claimed != canonical_sha256(unhashed)
        or interface not in INTERFACES
        or receipt.get("feature_semantics") != "physical_phase_clock_coordinates"
        or receipt.get("width") != INTERFACES[interface][0]
        or receipt.get("status")
        != "VERIFIED_STRUCTURAL_OOF_STITCH_NOT_SCIENTIFICALLY_ADMITTED"
        or receipt.get("artifact_files_rehashed") is not True
        or receipt.get("targets_read_for_stitching") is not False
        or receipt.get("blind_validation_claim") is not False
        or receipt.get("scientific_launch_authorized") is not False
        or _contains_forbidden_representation(receipt)
    ):
        raise ValueError("OOF stitch identity or physical interface differs")
    contract_path = Path(str(receipt["fold_contract_path"])).resolve(strict=True)
    if file_sha256(contract_path) != receipt.get("fold_contract_sha256"):
        raise ValueError("OOF stitch fold contract changed")
    expected_population, expected_groups, expected_tokens, _, expected_ordinals = (
        _fold_universe(contract_path)
    )
    fragments = receipt.get("fragments")
    if not isinstance(fragments, list) or len(fragments) != 5:
        raise ValueError("OOF stitch requires exactly five generation fragments")
    total = 0
    group_union: set[str] = set()
    token_union: set[str] = set()
    ordinal_union: set[int] = set()
    token_hashes: list[str] = []
    for rotation, fragment in enumerate(fragments):
        identity = load_rotation_identity(contract_path, rotation)
        generation = identity.roles["generation"]
        if (
            fragment.get("rotation") != rotation
            or fragment.get("generation_rows_sha256") != generation.rows_sha256
            or fragment.get("ordered_sample_token_sha256")
            != generation.ordered_sample_token_sha256
            or fragment.get("sample_token_set_sha256") != generation.sample_token_set_sha256
            or fragment.get("group_ids") != list(generation.group_ids)
            or int(fragment.get("population", -1)) != generation.population
        ):
            raise ValueError("OOF stitch fragment differs from exact generation rows")
        lineage_refs = fragment.get("endpoint_lineages")
        if not isinstance(lineage_refs, dict) or set(lineage_refs) != set(INTERFACES[interface][1]):
            raise ValueError("OOF stitch endpoints differ from its physical interface")
        for name, reference in lineage_refs.items():
            path = Path(str(reference["path"])).resolve(strict=True)
            if file_sha256(path) != reference.get("file_sha256"):
                raise ValueError("OOF stitch endpoint lineage changed")
            endpoint = verify_endpoint_lineage(path)
            if (
                endpoint.get("stage_id") != name
                or endpoint.get("rotation") != rotation
                or endpoint.get("lineage_sha256") != reference.get("lineage_sha256")
            ):
                raise ValueError("OOF stitch endpoint belongs to another stage or rotation")
        payload_path = Path(str(fragment["payload_path"])).resolve(strict=True)
        if file_sha256(payload_path) != fragment.get("payload_sha256"):
            raise ValueError("OOF stitch fragment payload changed")
        total += generation.population
        if group_union & set(generation.group_ids):
            raise ValueError("OOF stitch generation groups overlap")
        group_union.update(generation.group_ids)
        generation_rows = pd.read_parquet(
            generation.rows_path, columns=["sample_token", "source_ordinal"]
        )
        generation_tokens = set(generation_rows["sample_token"].astype(str))
        generation_ordinals = set(generation_rows["source_ordinal"].astype(int))
        if token_union & generation_tokens or ordinal_union & generation_ordinals:
            raise ValueError("OOF stitch generation tokens or ordinals overlap")
        token_union.update(generation_tokens)
        ordinal_union.update(generation_ordinals)
        token_hashes.append(generation.sample_token_set_sha256)
    if (
        total != int(receipt.get("population", -1))
        or total != expected_population
        or len(group_union) != int(receipt.get("groups", -1))
        or len(group_union) != expected_groups
        or token_union != expected_tokens
        or ordinal_union != expected_ordinals
        or receipt.get("generation_token_set_sha256")
        != set_identity_sha256(sorted(token_union))
        or receipt.get("generation_source_ordinal_set_sha256")
        != set_identity_sha256([str(value) for value in sorted(ordinal_union)])
        or receipt.get("generation_token_sets_sha256") != canonical_sha256(token_hashes)
    ):
        raise ValueError("OOF stitch does not cover TRAIN40 exactly once")
    return receipt


def build_stitch_receipt(
    *,
    fold_contract_path: Path,
    interface: str,
    fragment_payloads: Mapping[int, Path],
    endpoint_lineages: Mapping[int, Mapping[str, Path]],
) -> dict[str, Any]:
    """Bind five generation payloads to exact rows and temporary endpoint ancestry."""
    if interface not in INTERFACES or set(fragment_payloads) != set(range(5)) or set(
        endpoint_lineages
    ) != set(range(5)):
        raise ValueError("OOF stitch inputs require rotations 0..4 and a known physical interface")
    contract_path = fold_contract_path.resolve(strict=True)
    expected_population, expected_groups, expected_tokens, _, expected_ordinals = (
        _fold_universe(contract_path)
    )
    fragments: list[dict[str, Any]] = []
    token_hashes: list[str] = []
    population = 0
    groups: set[str] = set()
    for rotation in range(5):
        identity = load_rotation_identity(contract_path, rotation)
        generation = identity.roles["generation"]
        references: dict[str, dict[str, str]] = {}
        if set(endpoint_lineages[rotation]) != set(INTERFACES[interface][1]):
            raise ValueError("Stitch endpoint stages differ from requested interface")
        for name, raw_path in endpoint_lineages[rotation].items():
            path = raw_path.resolve(strict=True)
            endpoint = verify_endpoint_lineage(path)
            if endpoint["stage_id"] != name or endpoint["rotation"] != rotation:
                raise ValueError("Stitch endpoint lineage belongs to another fold")
            references[name] = {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "lineage_sha256": str(endpoint["lineage_sha256"]),
            }
        payload_path = fragment_payloads[rotation].resolve(strict=True)
        fragments.append(
            {
                "rotation": rotation,
                "generation_rows_sha256": generation.rows_sha256,
                "ordered_sample_token_sha256": generation.ordered_sample_token_sha256,
                "sample_token_set_sha256": generation.sample_token_set_sha256,
                "population": generation.population,
                "group_ids": list(generation.group_ids),
                "payload_path": str(payload_path),
                "payload_sha256": file_sha256(payload_path),
                "endpoint_lineages": references,
            }
        )
        population += generation.population
        groups.update(generation.group_ids)
        token_hashes.append(generation.sample_token_set_sha256)
    payload: dict[str, Any] = {
        "schema": STITCH_SCHEMA,
        "status": "VERIFIED_STRUCTURAL_OOF_STITCH_NOT_SCIENTIFICALLY_ADMITTED",
        "fold_contract_path": str(contract_path),
        "fold_contract_sha256": file_sha256(contract_path),
        "interface": interface,
        "feature_semantics": "physical_phase_clock_coordinates",
        "width": INTERFACES[interface][0],
        "population": population,
        "groups": len(groups),
        "generation_token_set_sha256": set_identity_sha256(sorted(expected_tokens)),
        "generation_source_ordinal_set_sha256": set_identity_sha256(
            [str(value) for value in sorted(expected_ordinals)]
        ),
        "generation_token_sets_sha256": canonical_sha256(token_hashes),
        "fragments": fragments,
        "targets_read_for_stitching": False,
        "blind_validation_claim": False,
        "artifact_files_rehashed": True,
        "scientific_launch_authorized": False,
    }
    result = {**payload, "stitch_sha256": canonical_sha256(payload)}
    if population != expected_population or len(groups) != expected_groups:
        raise ValueError("OOF stitch does not match the exact fold universe")
    return verify_stitch_receipt(result)


def build_deployment_transition(
    *,
    stage_id: str,
    training_stitch_path: Path,
    training_normalizer_sha256: str | None,
    deployment_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    """Declare, without conflating, OOF training features and full-fit deployment parents."""
    stitch_path = training_stitch_path.resolve(strict=True)
    stitch = verify_stitch_receipt(stitch_path)
    dependencies = {
        str(name): _digest(str(digest), name=f"deployment dependency {name}")
        for name, digest in sorted(deployment_dependencies.items())
    }
    if not dependencies:
        raise ValueError("Deployment transition requires explicit full-fit dependencies")
    payload: dict[str, Any] = {
        "schema": TRANSITION_SCHEMA,
        "status": "DECLARED_NOT_EXECUTED",
        "stage_id": stage_id,
        "training_feature_ancestry": {
            "stitch_path": str(stitch_path),
            "stitch_file_sha256": file_sha256(stitch_path),
            "stitch_payload_sha256": stitch["stitch_sha256"],
            "normalizer_sha256": (
                None
                if training_normalizer_sha256 is None
                else _digest(training_normalizer_sha256, name="training_normalizer_sha256")
            ),
        },
        "deployment_dependencies": dependencies,
        "ancestry_domains_are_distinct": True,
        "blind_validation_claim": False,
        "scientific_launch_authorized": False,
    }
    return {**payload, "transition_sha256": canonical_sha256(payload)}


__all__ = [
    "ENDPOINT_SCHEMA",
    "INTERFACES",
    "STITCH_SCHEMA",
    "TRANSITION_SCHEMA",
    "RoleIdentity",
    "RotationIdentity",
    "build_deployment_transition",
    "build_endpoint_lineage",
    "build_stitch_receipt",
    "canonical_sha256",
    "file_sha256",
    "load_rotation_identity",
    "ordered_identity_sha256",
    "set_identity_sha256",
    "verify_endpoint_lineage",
    "verify_stitch_receipt",
    "write_endpoint_lineage",
]
