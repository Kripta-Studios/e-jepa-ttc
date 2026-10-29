"""Fail-closed verification and target admission for independent TRAIN24 fits."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .population import (
    BLOCK_SCHEMA,
    FOLD_SCHEMA,
    IDENTITY_COLUMNS,
    LINEAGE_SCHEMA,
    POPULATION_SCHEMA,
    ROTATION_SCHEMA,
    SOURCE_SCHEMA,
    TEACHER_BYTES,
    TEACHER_FILE_SHA256,
    TEACHER_REPO,
    TEACHER_REVISION,
    TEACHER_SCHEMA,
    TEACHER_TENSOR_SHA256,
    _load_raw_identity,
    _verify_split,
    balanced_group_blocks,
    canonical_sha256,
    file_sha256,
    identity_sha256,
)

_ROLES = ("producer", "refiner", "generation")
_FIT_ROLES = ("producer", "refiner")
_FORBIDDEN_PARENT_MARKERS = (
    "operational/final_refit",
    "learned_h8",
    "h8_cache",
    "history_cache",
    "artifacts/rgb_port_20261008",
)
_DIGEST_CHARS = frozenset("0123456789abcdef")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("Receipt canonical identity changed")


def _digest(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _DIGEST_CHARS for character in text):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return text


def _role_rows(manifest_path: Path, manifest: Mapping[str, Any], role: str) -> pd.DataFrame:
    reference = manifest["roles"][role]
    path = (manifest_path.parent / str(reference["path"])).resolve(strict=True)
    if file_sha256(path) != reference.get("sha256"):
        raise ValueError(f"Rotation {manifest.get('rotation')} {role} rows changed")
    frame = pd.read_parquet(path)
    required = {"rotation", "rotation_role", *IDENTITY_COLUMNS}
    if set(frame.columns) != required:
        raise ValueError(f"Rotation {role} row schema differs")
    tokens = frame["sample_token"].astype(str).tolist()
    groups = set(frame["group_id"].astype(str))
    if (
        len(frame) != int(reference["queries"])
        or len(groups) != int(reference["groups"])
        or sorted(groups) != reference["group_ids"]
        or frame["sample_token"].astype(str).duplicated().any()
        or frame["fit_ordinal"].astype(int).duplicated().any()
        or set(frame["rotation"].astype(int)) != {int(manifest["rotation"])}
        or set(frame["rotation_role"].astype(str)) != {role}
        or identity_sha256(tokens) != reference["ordered_sample_token_sha256"]
        or identity_sha256(tokens, ordered=False) != reference["sample_token_set_sha256"]
    ):
        raise ValueError(f"Rotation {manifest.get('rotation')} {role} identity differs")
    return frame


def verify_independent_contract(fold_contract_path: Path) -> dict[str, Any]:
    """Rehash and verify the complete population, blocks, rotations, and lineage."""
    fold_path = fold_contract_path.resolve(strict=True)
    root = fold_path.parent
    fold = _json(fold_path)
    _verify_self_hash(fold)
    if (
        fold.get("schema") != FOLD_SCHEMA
        or fold.get("status") != "PASS_NOT_TRAINED"
        or fold.get("seed") != 7
        or fold.get("population") != {"queries": 53_047, "groups": 24}
        or fold.get("generation_coverage") != "EACH_TRAIN24_QUERY_EXACTLY_ONCE"
        or fold.get("group_role_frequency") != {"generation": 1, "refiner": 1, "producer": 4}
        or fold.get("targets_used_for_partitioning") is not False
        or fold.get("training_started") is not False
        or fold.get("optimizer_updates") != 0
    ):
        raise ValueError("Independent fold contract differs")

    source_path = root / "SOURCE_RECEIPT.json"
    population_receipt_path = root / "POPULATION.json"
    population_path = root / "population.parquet"
    blocks_path = root / "BLOCKS.json"
    lineage_path = root / "LINEAGE.json"
    teacher_path = root / "TEACHER_PARENT.json"
    source = _json(source_path)
    population_receipt = _json(population_receipt_path)
    blocks = _json(blocks_path)
    lineage = _json(lineage_path)
    teacher = _json(teacher_path)
    for value in (source, population_receipt, blocks, lineage, teacher):
        _verify_self_hash(value)
    if source.get("schema") != SOURCE_SCHEMA or source.get("ttc_values_read") is not False:
        raise ValueError("Raw source receipt differs")
    for name, reference in source.get("sources", {}).items():
        path = Path(str(reference.get("path", ""))).resolve(strict=True)
        if file_sha256(path) != reference.get("sha256"):
            raise ValueError(f"Raw source changed: {name}")
    if (
        population_receipt.get("schema") != POPULATION_SCHEMA
        or population_receipt.get("queries") != 53_047
        or population_receipt.get("groups") != 24
        or population_receipt.get("selection_entries") != 0
        or population_receipt.get("calibration_entries") != 0
        or population_receipt.get("targets_copied") is not False
        or population_receipt.get("source_receipt_sha256") != file_sha256(source_path)
        or population_receipt.get("population_sha256") != file_sha256(population_path)
        or fold.get("population_receipt_sha256") != file_sha256(population_receipt_path)
        or fold.get("blocks_sha256") != file_sha256(blocks_path)
        or blocks.get("schema") != BLOCK_SCHEMA
        or blocks.get("target_values_read") is not False
    ):
        raise ValueError("Population or block binding differs")
    population = pd.read_parquet(population_path)
    tokens = population["sample_token"].astype(str).tolist()
    groups = set(population["group_id"].astype(str))
    if (
        list(population.columns) != IDENTITY_COLUMNS
        or len(population) != 53_047
        or len(groups) != 24
        or population["sample_token"].astype(str).duplicated().any()
        or population["public_source_ordinal"].astype(int).duplicated().any()
        or population["fit_ordinal"].astype(int).tolist() != list(range(53_047))
        or identity_sha256(tokens) != population_receipt["ordered_sample_token_sha256"]
        or identity_sha256(tokens, ordered=False) != population_receipt["sample_token_set_sha256"]
    ):
        raise ValueError("Exact TRAIN24 population identity differs")

    source_paths = source["sources"]
    _, split_roles = _verify_split(Path(str(source_paths["split"]["path"])))
    raw_population, raw_counts, raw_history = _load_raw_identity(
        Path(str(source_paths["garl_data"]["path"])),
        Path(str(source_paths["garl_annotations"]["path"])),
        Path(str(source_paths["eap_data"]["path"])),
        split_roles,
    )
    if (
        raw_counts != population_receipt["split_query_counts"]
        or raw_history != population_receipt["history_audit"]
        or not raw_population[IDENTITY_COLUMNS].equals(population[IDENTITY_COLUMNS])
        or identity_sha256(population["public_source_ordinal"].tolist())
        != population_receipt["ordered_public_source_ordinal_sha256"]
    ):
        raise ValueError("Population does not exactly rederive from raw metadata and split")
    raw_group_counts = raw_population.groupby("group_id", sort=True).size().astype(int).to_dict()
    expected_blocks = balanced_group_blocks(raw_group_counts)
    declared_blocks = blocks.get("blocks")
    if not isinstance(declared_blocks, dict):
        raise ValueError("Block declaration differs")
    for index, expected_groups in enumerate(expected_blocks):
        declared = declared_blocks.get(f"B{index}")
        if declared != {
            "groups": expected_groups,
            "queries": sum(raw_group_counts[group] for group in expected_groups),
        }:
            raise ValueError("Blocks do not rederive from label-free raw group counts")

    rotations = fold.get("rotations")
    if not isinstance(rotations, list) or len(rotations) != 6:
        raise ValueError("Expected exactly six rotations")
    generated_tokens: set[str] = set()
    generated_groups: set[str] = set()
    role_frequency = {group: {role: 0 for role in _ROLES} for group in groups}
    for rotation, reference in enumerate(rotations):
        if reference.get("rotation") != rotation:
            raise ValueError("Rotation order differs")
        manifest_path = (root / str(reference["path"])).resolve(strict=True)
        if file_sha256(manifest_path) != reference.get("sha256"):
            raise ValueError("Rotation manifest changed")
        manifest = _json(manifest_path)
        _verify_self_hash(manifest)
        if (
            manifest.get("schema") != ROTATION_SCHEMA
            or manifest.get("rotation") != rotation
            or manifest.get("partition") != "16/4/4_DISJOINT_GROUPS_COVER_TRAIN24"
            or manifest.get("generation_ancestry_excludes_group") is not True
            or manifest.get("targets_copied") is not False
        ):
            raise ValueError("Rotation manifest differs")
        frames = {role: _role_rows(manifest_path, manifest, role) for role in _ROLES}
        role_groups = {role: set(frame["group_id"].astype(str)) for role, frame in frames.items()}
        role_tokens = {
            role: set(frame["sample_token"].astype(str)) for role, frame in frames.items()
        }
        if tuple(len(role_groups[role]) for role in _ROLES) != (16, 4, 4):
            raise ValueError("Rotation is not the required 16/4/4 partition")
        if any(
            role_groups[left] & role_groups[right] or role_tokens[left] & role_tokens[right]
            for left, right in (
                ("producer", "refiner"),
                ("producer", "generation"),
                ("refiner", "generation"),
            )
        ):
            raise ValueError("Rotation roles overlap")
        if set().union(*role_groups.values()) != groups or set().union(
            *role_tokens.values()
        ) != set(tokens):
            raise ValueError("Rotation roles do not cover TRAIN24")
        expected_role_groups = {
            "generation": set(expected_blocks[rotation]),
            "refiner": set(expected_blocks[(rotation + 1) % 6]),
            "producer": groups
            - set(expected_blocks[rotation])
            - set(expected_blocks[(rotation + 1) % 6]),
        }
        for role in _ROLES:
            expected_frame = population.loc[
                population["group_id"].astype(str).isin(sorted(expected_role_groups[role]))
            ].sort_values("fit_ordinal", kind="stable")
            observed_identity = frames[role][IDENTITY_COLUMNS].reset_index(drop=True)
            expected_identity = expected_frame[IDENTITY_COLUMNS].reset_index(drop=True)
            if role_groups[role] != expected_role_groups[role] or not observed_identity.equals(
                expected_identity
            ):
                raise ValueError("Rotation role identities or B_i/B_i+1 law differ")
        if (
            generated_tokens & role_tokens["generation"]
            or generated_groups & role_groups["generation"]
        ):
            raise ValueError("OOF generation roles overlap")
        generated_tokens |= role_tokens["generation"]
        generated_groups |= role_groups["generation"]
        for role in _ROLES:
            for group in role_groups[role]:
                role_frequency[group][role] += 1
    if generated_tokens != set(tokens) or generated_groups != groups:
        raise ValueError("OOF generation does not cover TRAIN24 exactly once")
    if any(
        counts != {"producer": 4, "refiner": 1, "generation": 1}
        for counts in role_frequency.values()
    ):
        raise ValueError("A group does not have the exact six-rotation role frequency")
    if (
        lineage.get("schema") != LINEAGE_SCHEMA
        or lineage.get("source_receipt_sha256") != file_sha256(source_path)
        or lineage.get("population_receipt_sha256") != file_sha256(population_receipt_path)
        or lineage.get("blocks_sha256") != file_sha256(blocks_path)
        or lineage.get("fold_contract_sha256") != file_sha256(fold_path)
        or lineage.get("teacher_parent_receipt_sha256") != file_sha256(teacher_path)
        or lineage.get("historical_learned_parents_allowed") is not False
        or lineage.get("teacher_population_whitelist") != "TRAIN24_ONLY"
        or lineage.get("models_trained") != 0
    ):
        raise ValueError("Population lineage binding differs")
    if (
        teacher.get("schema") != TEACHER_SCHEMA
        or teacher.get("status") != "AVAILABLE_FROZEN_NOT_EXTRACTED_FOR_INDEPENDENT_REFIT"
        or teacher.get("repository") != TEACHER_REPO
        or teacher.get("revision") != TEACHER_REVISION
        or teacher.get("weights_bytes") != TEACHER_BYTES
        or teacher.get("weights_file_sha256") != TEACHER_FILE_SHA256
        or teacher.get("ordered_state_tensor_sha256") != TEACHER_TENSOR_SHA256
        or teacher.get("fold_contract_sha256") != file_sha256(fold_path)
        or teacher.get("query_token_set_sha256") != population_receipt["sample_token_set_sha256"]
        or teacher.get("teacher_features_materialized") is not False
    ):
        raise ValueError("Frozen teacher parent receipt differs")
    teacher_weights = Path(str(teacher.get("weights_path", ""))).resolve(strict=True)
    if (
        teacher_weights.stat().st_size != TEACHER_BYTES
        or file_sha256(teacher_weights) != TEACHER_FILE_SHA256
    ):
        raise ValueError("Frozen teacher parent bytes changed")
    return {
        "status": "PASS",
        "queries": len(population),
        "groups": len(groups),
        "rotations": 6,
        "fold_contract_sha256": file_sha256(fold_path),
        "population_sha256": file_sha256(population_path),
        "sample_token_set_sha256": identity_sha256(tokens, ordered=False),
    }


def _load_rotation_role_unverified(
    fold_contract_path: Path, *, rotation: int, role: str
) -> pd.DataFrame:
    if role not in _ROLES or rotation not in range(6):
        raise ValueError("Unknown independent rotation role")
    fold_path = fold_contract_path.resolve(strict=True)
    fold = _json(fold_path)
    reference = fold["rotations"][rotation]
    manifest_path = (fold_path.parent / str(reference["path"])).resolve(strict=True)
    return _role_rows(manifest_path, _json(manifest_path), role)


def load_rotation_role(fold_contract_path: Path, *, rotation: int, role: str) -> pd.DataFrame:
    """Load one exact verified role after validating the complete contract."""
    verify_independent_contract(fold_contract_path)
    return _load_rotation_role_unverified(fold_contract_path, rotation=rotation, role=role)


def join_supervised_targets(
    fold_contract_path: Path,
    *,
    rotation: int,
    fit_role: str,
    annotations_path: Path,
) -> pd.DataFrame:
    """Join TTC only to a fit's own producer/refiner rows after contract verification.

    Generation rows are never accepted. The returned frame is in ``fit_ordinal``
    order and is not written into any label-free fold artifact.
    """
    if fit_role not in _FIT_ROLES:
        raise ValueError("Supervised targets are admitted only for own producer/refiner fit roles")
    rows = load_rotation_role(fold_contract_path, rotation=rotation, role=fit_role)
    source = _json(fold_contract_path.resolve(strict=True).parent / "SOURCE_RECEIPT.json")
    annotation_ref = source["sources"]["garl_annotations"]
    if (
        annotations_path.resolve(strict=True)
        != Path(str(annotation_ref["path"])).resolve(strict=True)
        or file_sha256(annotations_path) != annotation_ref["sha256"]
    ):
        raise ValueError("Supervision source differs from the population-bound annotation metadata")
    annotations = pq.read_table(
        annotations_path,
        columns=[
            "sequence_id",
            "sample_token",
            "track_id",
            "public_track_id",
            "timestamp_us",
            "ttc",
        ],
        filters=[("sequence_id", "in", sorted(set(rows["group_id"].astype(str))))],
    ).to_pandas()
    keys = ["sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us"]
    if annotations["sample_token"].astype(str).duplicated().any():
        raise ValueError("Supervision contains duplicate query tokens")
    joined = rows.merge(annotations, on=keys, how="left", validate="one_to_one")
    if bool(joined["ttc"].isna().any()) or len(joined) != len(rows):
        raise ValueError("Supervision join is incomplete")
    return joined.sort_values("fit_ordinal", kind="stable").reset_index(drop=True)


def validate_fit_parents(
    fold_contract_path: Path,
    *,
    rotation: int,
    fit_role: str,
    component_id: str,
    parents: Sequence[Mapping[str, Any]],
) -> None:
    """Verify physical parent artifacts and the bounded component dependency graph."""
    dependencies = {
        "E_A5": ("producer", {"teacher"}),
        "E_C2F": ("producer", {"teacher"}),
        "R_A5": ("producer", {"teacher"}),
        "R_C2F": ("producer", {"teacher"}),
        "PAIR_E": ("producer", {"E_A5"}),
        "PAIR_R": ("producer", {"R_A5"}),
        "E_CTX": ("refiner", {"E_A5", "E_C2F", "PAIR_E"}),
        "R_CTX": ("refiner", {"R_A5", "R_C2F", "PAIR_R"}),
    }
    if component_id not in dependencies or fit_role != dependencies[component_id][0]:
        raise ValueError("Component and supervised fit role differ from the supported graph")
    verify_independent_contract(fold_contract_path)
    own = _load_rotation_role_unverified(fold_contract_path, rotation=rotation, role=fit_role)
    producer = _load_rotation_role_unverified(
        fold_contract_path, rotation=rotation, role="producer"
    )
    generation = _load_rotation_role_unverified(
        fold_contract_path, rotation=rotation, role="generation"
    )
    own_groups = set(own["group_id"].astype(str))
    producer_groups = set(producer["group_id"].astype(str))
    generation_groups = set(generation["group_id"].astype(str))
    contract_sha = file_sha256(fold_contract_path)
    expected_stages = dependencies[component_id][1]
    observed_stages: set[str] = set()
    has_unadmitted_local_fit = False
    for parent in parents:
        receipt_path = Path(str(parent.get("receipt_path", ""))).resolve(strict=True)
        receipt_file_sha = file_sha256(receipt_path)
        if receipt_file_sha != _digest(parent.get("receipt_sha256"), name="receipt sha256"):
            raise ValueError("Parent receipt bytes differ from their declared SHA-256")
        receipt = _json(receipt_path)
        _verify_self_hash(receipt)
        path_text = (
            str(receipt.get("artifact_path", receipt.get("weights_path", "")))
            .replace("\\", "/")
            .lower()
        )
        if any(marker in path_text for marker in _FORBIDDEN_PARENT_MARKERS):
            raise ValueError("Historical checkpoint/normalizer/learned-cache parent is forbidden")
        kind = receipt.get("kind")
        if kind == "frozen_dinov3_teacher_weights":
            if "teacher" in observed_stages:
                raise ValueError("Duplicate direct parent stage")
            if (
                receipt_path
                != (fold_contract_path.resolve(strict=True).parent / "TEACHER_PARENT.json")
                or receipt.get("schema") != TEACHER_SCHEMA
                or receipt.get("repository") != TEACHER_REPO
                or receipt.get("revision") != TEACHER_REVISION
                or receipt.get("weights_file_sha256") != TEACHER_FILE_SHA256
                or receipt.get("ordered_state_tensor_sha256") != TEACHER_TENSOR_SHA256
                or receipt.get("fold_contract_sha256") != contract_sha
            ):
                raise ValueError("DINOv3 teacher receipt differs from the approved physical parent")
            weights = Path(str(receipt["weights_path"])).resolve(strict=True)
            if (
                weights.stat().st_size != TEACHER_BYTES
                or file_sha256(weights) != TEACHER_FILE_SHA256
            ):
                raise ValueError("DINOv3 teacher physical bytes changed")
            observed_stages.add("teacher")
            continue
        if kind != "independent_phase_fit":
            raise ValueError("Unknown learned parent kind")
        if (
            receipt.get("schema") != "independent_phase_fit_parent_v1"
            or receipt.get("status") != "COMPLETE_VERIFIED_INDEPENDENT_FIT"
        ):
            raise ValueError("Independent fit receipt schema/status differs")
        artifact = Path(str(receipt.get("artifact_path", ""))).resolve(strict=True)
        artifact_sha = file_sha256(artifact)
        parent_groups = {str(item) for item in receipt.get("train_group_ids", [])}
        parent_stage = str(receipt.get("stage_id", ""))
        if parent_stage in observed_stages:
            raise ValueError("Duplicate direct parent stage")
        expected_tokens = producer["sample_token"].astype(str).tolist()
        if (
            receipt.get("artifact_sha256") != artifact_sha
            or receipt.get("fold_contract_sha256") != contract_sha
            or receipt.get("rotation") != rotation
            or receipt.get("train_role") != "producer"
            or parent_groups != producer_groups
            or parent_groups & generation_groups
            or receipt.get("train_token_set_sha256")
            != identity_sha256(expected_tokens, ordered=False)
            or receipt.get("generation_group_ids") != sorted(generation_groups)
            or receipt.get("generation_excluded") is not True
        ):
            raise ValueError("Independent parent has cross-role or generation ancestry")
        observed_stages.add(parent_stage)
        has_unadmitted_local_fit = True
    if observed_stages != expected_stages:
        raise ValueError("Direct parents differ from the explicit component dependency graph")
    if own_groups & generation_groups:
        raise AssertionError("Child supervision overlaps its generation groups")
    if has_unadmitted_local_fit:
        raise RuntimeError(
            "Independent learned parents are fail-closed until the fitter emits an admitted "
            "checkpoint-internal identity and training receipt schema"
        )


__all__ = [
    "join_supervised_targets",
    "load_rotation_role",
    "validate_fit_parents",
    "verify_independent_contract",
]
