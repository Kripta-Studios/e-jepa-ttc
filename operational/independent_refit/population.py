"""Build the label-free independent TRAIN24 population and six OOF rotations.

Only raw metadata identities are read. TTC values, sensor payloads, model weights,
and historical TRAIN40 learned artifacts are outside this module's contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 7
EXPECTED_SPLIT_SHA256 = "91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903"
EXPECTED_EAP_SHA256 = "69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314"
EXPECTED_COUNTS = {"train": 53_047, "selection": 18_594, "calibration": 17_103}
POPULATION_SCHEMA = "independent_phase_train24_population_v2"
BLOCK_SCHEMA = "independent_phase_six_blocks_v2"
ROTATION_SCHEMA = "independent_phase_rotation_roles_v2"
FOLD_SCHEMA = "independent_phase_oof_fold_contract_v2"
SOURCE_SCHEMA = "independent_phase_raw_metadata_sources_v2"
LINEAGE_SCHEMA = "independent_phase_population_lineage_v2"
TEACHER_SCHEMA = "independent_phase_frozen_teacher_parent_v1"
TEACHER_REPO = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
TEACHER_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
TEACHER_FILE_SHA256 = "8bab82eaa133954d0557bebf70416e0b667fa417a19ea47b697683b34190633d"
TEACHER_TENSOR_SHA256 = "96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628"
TEACHER_BYTES = 784_957_976

IDENTITY_COLUMNS = [
    "public_source_ordinal",
    "fit_ordinal",
    "sample_token",
    "group_id",
    "sequence_id",
    "track_id",
    "public_track_id",
    "timestamp_us",
]

_GARL_DATA_SCHEMA = {
    "sequence_id": "string",
    "sample_token": "string",
    "track_id": "string",
    "public_track_id": "string",
    "timestamp_us": "int64",
    "seq_name": "string",
    "frame_timestamps_us": "list<element: int64>",
    "rgb_shard_paths": "list<element: string>",
    "rgb_member_paths": "list<element: string>",
    "events_path": "string",
    "event_windows_us": "list<element: list<element: int64>>",
    "image_paths": "list<element: string>",
    "boxes_xyxy": "list<element: list<element: int64>>",
    "mask_paths": "list<element: string>",
}
_GARL_ANNOTATION_SCHEMA = {
    "sequence_id": "string",
    "sample_token": "string",
    "track_id": "string",
    "public_track_id": "string",
    "timestamp_us": "int64",
    "ttc": "double",
    "frame_ttc": "list<element: double>",
    "box3d_h": "double",
    "box3d_Fcam": "list<element: list<element: list<element: double>>>",
}
_EAP_SCHEMA = {
    "sample_token": "string",
    "split": "string",
    "sequence_id": "string",
    "rgb_shard_path": "string",
    "rgb_member_path": "string",
    "events_path": "string",
    "labels_path": "string",
    "rgb_exposure_start_timestamp_us": "int64",
    "rgb_exposure_end_timestamp_us": "int64",
    "K_event": "list<element: list<element: double>>",
    "T_event_ego": "list<element: list<element: double>>",
}


def file_sha256(path: Path) -> str:
    """Return SHA-256 for an existing regular file."""
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash strict canonical JSON."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def identity_sha256(values: Sequence[object], *, ordered: bool = True) -> str:
    """Hash an identity vector without delimiter ambiguity."""
    strings = [str(value) for value in values]
    if not ordered:
        if len(strings) != len(set(strings)):
            raise ValueError("Identity set contains duplicates")
        strings.sort()
    return canonical_sha256(strings)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_schema(path: Path, expected: Mapping[str, str]) -> pq.ParquetFile:
    parquet = pq.ParquetFile(path.resolve(strict=True))
    observed = {field.name: str(field.type) for field in parquet.schema_arrow}
    if observed != dict(expected):
        raise ValueError(f"Unknown metadata schema for {path}: {observed}")
    return parquet


def _verify_split(path: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    if file_sha256(path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Detection split bytes differ from the pinned SHA-256")
    split = _read_json(path)
    assignments = split.get("assignments")
    if (
        split.get("schema") != "eap_detection_sequence_split_v1"
        or split.get("seed") != SEED
        or split.get("media_sha256") != EXPECTED_EAP_SHA256
        or not isinstance(assignments, dict)
        or set(assignments) != set(EXPECTED_COUNTS)
    ):
        raise ValueError("Detection split contract differs")
    roles = {name: {str(item) for item in assignments[name]} for name in EXPECTED_COUNTS}
    if tuple(map(len, roles.values())) != (24, 8, 8):
        raise ValueError("Detection split is not 24/8/8 groups")
    if any(
        roles[left] & roles[right]
        for left, right in (
            ("train", "selection"),
            ("train", "calibration"),
            ("selection", "calibration"),
        )
    ):
        raise ValueError("Detection split roles overlap")
    return split, roles


def _same_sequence_path(sequence_id: str, path: object) -> bool:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    return sequence_id in parts


def _load_raw_identity(
    garl_data_path: Path,
    garl_annotations_path: Path,
    eap_data_path: Path,
    roles: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    data_file = _verify_schema(garl_data_path, _GARL_DATA_SCHEMA)
    annotation_file = _verify_schema(garl_annotations_path, _GARL_ANNOTATION_SCHEMA)
    eap_file = _verify_schema(eap_data_path, _EAP_SCHEMA)
    if data_file.metadata.num_rows != 88_744 or annotation_file.metadata.num_rows != 88_744:
        raise ValueError("Garl raw metadata is not the 88,744-query TRAIN40 universe")
    if eap_file.metadata.num_rows != 118_247:
        raise ValueError("eAP raw metadata frame count changed")

    data_columns = [
        "sequence_id",
        "sample_token",
        "track_id",
        "public_track_id",
        "timestamp_us",
        "frame_timestamps_us",
        "rgb_shard_paths",
    ]
    data = pq.read_table(garl_data_path, columns=data_columns).to_pandas()
    annotations = pq.read_table(
        garl_annotations_path,
        columns=["sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us"],
    ).to_pandas()
    identity = ["sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us"]
    if not data[identity].equals(annotations[identity]):
        raise ValueError("Garl data and annotation identity/order differ")
    if data["sample_token"].isna().any() or data["sample_token"].astype(str).duplicated().any():
        raise ValueError("Raw query tokens are null or duplicated")
    if data["sequence_id"].nunique() != 40:
        raise ValueError("Garl raw metadata does not contain exactly 40 groups")

    all_groups = set().union(*roles.values())
    if set(data["sequence_id"].astype(str)) != all_groups:
        raise ValueError("Garl groups differ from the pinned split universe")
    eap = pq.read_table(eap_data_path, columns=["sequence_id", "sample_token"]).to_pandas()
    if (
        eap["sample_token"].astype(str).duplicated().any()
        or set(eap["sequence_id"].astype(str)) != all_groups
    ):
        raise ValueError("eAP public identity universe differs")

    future_history = 0
    cross_sequence_history = 0
    history_slots = 0
    for row in data.itertuples(index=False):
        timestamps = [] if row.frame_timestamps_us is None else list(row.frame_timestamps_us)
        shards = [] if row.rgb_shard_paths is None else list(row.rgb_shard_paths)
        if len(timestamps) != len(shards):
            raise ValueError(f"History metadata lengths differ for {row.sample_token}")
        history_slots += len(timestamps)
        future_history += sum(int(value) > int(row.timestamp_us) for value in timestamps)
        cross_sequence_history += sum(
            not _same_sequence_path(str(row.sequence_id), value) for value in shards
        )
    if future_history or cross_sequence_history:
        raise ValueError("Raw history metadata contains future or cross-sequence inputs")

    data.insert(0, "public_source_ordinal", pd.Series(range(len(data)), dtype="int64"))
    role_counts = {
        role: int(data["sequence_id"].astype(str).isin(groups).sum())
        for role, groups in roles.items()
    }
    if role_counts != EXPECTED_COUNTS:
        raise ValueError(f"Raw split query counts changed: {role_counts}")
    train = data.loc[
        data["sequence_id"].astype(str).isin(roles["train"]), identity + ["public_source_ordinal"]
    ].copy()
    train = train.sort_values("public_source_ordinal", kind="stable").reset_index(drop=True)
    train.insert(1, "fit_ordinal", pd.Series(range(len(train)), dtype="int64"))
    train["group_id"] = train["sequence_id"].astype(str)
    train = train[IDENTITY_COLUMNS]
    audit = {
        "history_slots_checked": history_slots,
        "future_history_entries": future_history,
        "cross_sequence_history_entries": cross_sequence_history,
        "annotation_values_read": False,
        "sensor_payloads_read": False,
        "historical_train40_learned_artifacts_read": False,
    }
    return train, role_counts, audit


def balanced_group_blocks(group_counts: Mapping[str, int], *, seed: int = SEED) -> list[list[str]]:
    """Assign 24 groups to six capacity-four blocks using seeded greedy balancing.

    Groups are visited by descending query count. Equal counts use
    ``sha256(f"{seed}:{group_id}")`` then the group ID. Each group is placed in
    the non-full block with the lowest current query total, then lowest block
    cardinality, then block number.
    """
    if len(group_counts) != 24 or any(int(value) <= 0 for value in group_counts.values()):
        raise ValueError("Greedy folding requires exactly 24 positive group counts")
    ordered = sorted(
        ((str(group), int(count)) for group, count in group_counts.items()),
        key=lambda item: (
            -item[1],
            hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )
    blocks: list[list[str]] = [[] for _ in range(6)]
    totals = [0] * 6
    for group, count in ordered:
        candidates = [index for index, block in enumerate(blocks) if len(block) < 4]
        chosen = min(candidates, key=lambda index: (totals[index], len(blocks[index]), index))
        blocks[chosen].append(group)
        totals[chosen] += count
    if any(len(block) != 4 for block in blocks):
        raise AssertionError("Greedy block capacity invariant failed")
    return [sorted(block) for block in blocks]


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", write_statistics=True)


def _role_frame(
    population: pd.DataFrame, rotation: int, role: str, groups: set[str]
) -> pd.DataFrame:
    selected = population.loc[population["group_id"].astype(str).isin(sorted(groups))].copy()
    selected.insert(0, "rotation", rotation)
    selected.insert(1, "rotation_role", role)
    return selected.sort_values("fit_ordinal", kind="stable").reset_index(drop=True)


def build_independent_population(
    *,
    split_path: Path,
    garl_data_path: Path,
    garl_annotations_path: Path,
    eap_data_path: Path,
    teacher_weights_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize a write-once label-free TRAIN24 population and OOF contract."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    split, split_roles = _verify_split(split_path)
    if file_sha256(eap_data_path) != EXPECTED_EAP_SHA256:
        raise ValueError("eAP raw metadata bytes differ from the pinned SHA-256")
    source_receipt: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "status": "VERIFIED_RAW_METADATA_ONLY",
        "sources": {
            "split": {"path": str(split_path.resolve()), "sha256": file_sha256(split_path)},
            "garl_data": {
                "path": str(garl_data_path.resolve()),
                "sha256": file_sha256(garl_data_path),
            },
            "garl_annotations": {
                "path": str(garl_annotations_path.resolve()),
                "sha256": file_sha256(garl_annotations_path),
            },
            "eap_data": {
                "path": str(eap_data_path.resolve()),
                "sha256": file_sha256(eap_data_path),
            },
        },
        "garl_hashes_observed_not_predeclared": True,
        "ttc_values_read": False,
    }
    source_receipt["receipt_sha256"] = canonical_sha256(source_receipt)
    _write_json(output_dir / "SOURCE_RECEIPT.json", source_receipt)

    population, role_counts, history_audit = _load_raw_identity(
        garl_data_path, garl_annotations_path, eap_data_path, split_roles
    )
    population_path = output_dir / "population.parquet"
    _write_parquet(population_path, population)
    population_receipt: dict[str, Any] = {
        "schema": POPULATION_SCHEMA,
        "status": "VERIFIED_LABEL_FREE_TRAIN24",
        "source_receipt_sha256": file_sha256(output_dir / "SOURCE_RECEIPT.json"),
        "population_path": population_path.name,
        "population_sha256": file_sha256(population_path),
        "queries": len(population),
        "groups": len(set(population["group_id"].astype(str))),
        "split_query_counts": role_counts,
        "selection_entries": 0,
        "calibration_entries": 0,
        "ordered_sample_token_sha256": identity_sha256(population["sample_token"].tolist()),
        "sample_token_set_sha256": identity_sha256(
            population["sample_token"].tolist(), ordered=False
        ),
        "ordered_public_source_ordinal_sha256": identity_sha256(
            population["public_source_ordinal"].tolist()
        ),
        "fit_ordinals": "CONTIGUOUS_0_TO_53046",
        "identity_columns": IDENTITY_COLUMNS,
        "history_audit": history_audit,
        "targets_copied": False,
    }
    population_receipt["receipt_sha256"] = canonical_sha256(population_receipt)
    _write_json(output_dir / "POPULATION.json", population_receipt)

    counts = population.groupby("group_id", sort=True).size().astype(int).to_dict()
    blocks = balanced_group_blocks(counts)
    block_payload: dict[str, Any] = {
        "schema": BLOCK_SCHEMA,
        "seed": SEED,
        "algorithm": "descending_count_seeded_sha256_tie_then_greedy_min_total_capacity_four",
        "target_values_read": False,
        "blocks": {
            f"B{index}": {
                "groups": block,
                "queries": sum(counts[group] for group in block),
            }
            for index, block in enumerate(blocks)
        },
    }
    block_payload["receipt_sha256"] = canonical_sha256(block_payload)
    _write_json(output_dir / "BLOCKS.json", block_payload)

    rotations: list[dict[str, Any]] = []
    generated_tokens: set[str] = set()
    generated_groups: set[str] = set()
    for rotation in range(6):
        role_groups = {
            "generation": set(blocks[rotation]),
            "refiner": set(blocks[(rotation + 1) % 6]),
            "producer": set().union(
                *(
                    set(blocks[index])
                    for index in range(6)
                    if index not in {rotation, (rotation + 1) % 6}
                )
            ),
        }
        rotation_dir = output_dir / "folds" / f"rotation_{rotation}"
        references: dict[str, Any] = {}
        role_tokens: dict[str, set[str]] = {}
        for role in ("producer", "refiner", "generation"):
            frame = _role_frame(population, rotation, role, role_groups[role])
            path = rotation_dir / f"{role}.parquet"
            _write_parquet(path, frame)
            tokens = frame["sample_token"].astype(str).tolist()
            role_tokens[role] = set(tokens)
            references[role] = {
                "path": path.name,
                "sha256": file_sha256(path),
                "queries": len(frame),
                "groups": len(role_groups[role]),
                "group_ids": sorted(role_groups[role]),
                "ordered_sample_token_sha256": identity_sha256(tokens),
                "sample_token_set_sha256": identity_sha256(tokens, ordered=False),
            }
        if any(
            role_tokens[left] & role_tokens[right]
            for left, right in (
                ("producer", "refiner"),
                ("producer", "generation"),
                ("refiner", "generation"),
            )
        ):
            raise AssertionError("Rotation roles overlap")
        if (
            generated_tokens & role_tokens["generation"]
            or generated_groups & role_groups["generation"]
        ):
            raise AssertionError("OOF generation coverage overlaps")
        generated_tokens |= role_tokens["generation"]
        generated_groups |= role_groups["generation"]
        manifest: dict[str, Any] = {
            "schema": ROTATION_SCHEMA,
            "status": "VERIFIED_LABEL_FREE_ROLES",
            "rotation": rotation,
            "roles": references,
            "partition": "16/4/4_DISJOINT_GROUPS_COVER_TRAIN24",
            "generation_ancestry_excludes_group": True,
            "targets_copied": False,
        }
        manifest["receipt_sha256"] = canonical_sha256(manifest)
        manifest_path = rotation_dir / "MANIFEST.json"
        _write_json(manifest_path, manifest)
        rotations.append(
            {
                "rotation": rotation,
                "path": str(manifest_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": file_sha256(manifest_path),
            }
        )
    population_tokens = set(population["sample_token"].astype(str))
    if generated_tokens != population_tokens or generated_groups != set(counts):
        raise AssertionError("OOF generation folds do not cover TRAIN24 exactly once")

    fold_contract: dict[str, Any] = {
        "schema": FOLD_SCHEMA,
        "status": "PASS_NOT_TRAINED",
        "seed": SEED,
        "population": {"queries": 53_047, "groups": 24},
        "population_receipt_sha256": file_sha256(output_dir / "POPULATION.json"),
        "blocks_sha256": file_sha256(output_dir / "BLOCKS.json"),
        "rotations": rotations,
        "generation_coverage": "EACH_TRAIN24_QUERY_EXACTLY_ONCE",
        "group_role_frequency": {"generation": 1, "refiner": 1, "producer": 4},
        "targets_used_for_partitioning": False,
        "training_started": False,
        "optimizer_updates": 0,
    }
    fold_contract["receipt_sha256"] = canonical_sha256(fold_contract)
    _write_json(output_dir / "FOLD_CONTRACT.json", fold_contract)
    teacher_path = teacher_weights_path.resolve(strict=True)
    if teacher_path.stat().st_size != TEACHER_BYTES:
        raise ValueError("Frozen DINOv3 teacher byte size differs")
    if file_sha256(teacher_path) != TEACHER_FILE_SHA256:
        raise ValueError("Frozen DINOv3 teacher file SHA-256 differs")
    teacher: dict[str, Any] = {
        "schema": TEACHER_SCHEMA,
        "status": "AVAILABLE_FROZEN_NOT_EXTRACTED_FOR_INDEPENDENT_REFIT",
        "kind": "frozen_dinov3_teacher_weights",
        "repository": TEACHER_REPO,
        "revision": TEACHER_REVISION,
        "weights_path": str(teacher_path),
        "weights_bytes": TEACHER_BYTES,
        "weights_file_sha256": TEACHER_FILE_SHA256,
        "ordered_state_tensor_sha256": TEACHER_TENSOR_SHA256,
        "fold_contract_sha256": file_sha256(output_dir / "FOLD_CONTRACT.json"),
        "query_whitelist": "TRAIN24_ONLY",
        "queries": 53_047,
        "query_token_set_sha256": population_receipt["sample_token_set_sha256"],
        "teacher_features_materialized": False,
        "optimizer_updates": 0,
    }
    teacher["receipt_sha256"] = canonical_sha256(teacher)
    _write_json(output_dir / "TEACHER_PARENT.json", teacher)
    lineage: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "status": "VERIFIED_POPULATION_AND_FOLDS_ONLY",
        "source_receipt_sha256": file_sha256(output_dir / "SOURCE_RECEIPT.json"),
        "population_receipt_sha256": file_sha256(output_dir / "POPULATION.json"),
        "blocks_sha256": file_sha256(output_dir / "BLOCKS.json"),
        "fold_contract_sha256": file_sha256(output_dir / "FOLD_CONTRACT.json"),
        "teacher_parent_receipt_sha256": file_sha256(output_dir / "TEACHER_PARENT.json"),
        "fit_parent_policy": "INDEPENDENT_LINEAGE_OR_FROZEN_DINOV3_TEACHER_ONLY",
        "historical_learned_parents_allowed": False,
        "teacher_population_whitelist": "TRAIN24_ONLY",
        "models_trained": 0,
    }
    lineage["receipt_sha256"] = canonical_sha256(lineage)
    _write_json(output_dir / "LINEAGE.json", lineage)

    from .lineage import verify_independent_contract

    return verify_independent_contract(output_dir / "FOLD_CONTRACT.json")


__all__ = [
    "EXPECTED_COUNTS",
    "EXPECTED_EAP_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "IDENTITY_COLUMNS",
    "SEED",
    "balanced_group_blocks",
    "build_independent_population",
    "canonical_sha256",
    "file_sha256",
    "identity_sha256",
]
