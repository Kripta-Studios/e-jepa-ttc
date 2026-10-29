"""Read-only admission checks for the final TRAIN40 refit.

The preflight deliberately avoids loading sensor payloads, TTC targets, model
weights, or test data.  Large cache shards are checked by manifest structure,
presence, and byte size; their historical content-hash admission is preserved
and reported instead of being repeated implicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_V12_REPO = "e-jepa-ttc-v12-efficient-context"
_V13_REPO = "e-jepa-ttc-v13-rgb-port"
_RUN12 = Path("artifacts/train40_system_20261005")
_RUN13 = Path("artifacts/rgb_port_20261008")
_ROLES = ("P", "H", "V")
_EVENT_FITS = ("a5_seed7", "c2f_seed7")


def _digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _source(workspace: Path, path: Path) -> dict[str, Any]:
    """Describe an internal source without embedding a machine-specific path."""
    try:
        display = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Internal source is outside workspace: {path}") from error
    return {"path": display, "sha256": _digest(path), "bytes": path.stat().st_size}


def _validate_shard_cache(
    manifest: dict[str, Any], *, label: str, expected_rows: int
) -> dict[str, Any]:
    if manifest.get("status") != "COMPLETE_VERIFIED":
        raise ValueError(f"{label} manifest is not COMPLETE_VERIFIED")
    if int(manifest.get("row_count", -1)) != expected_rows:
        raise ValueError(f"{label} row count does not match TRAIN40")
    shards = manifest.get("ordered_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{label} has no ordered shards")
    cache_dir = Path(str(manifest.get("directory", "")))
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"{label} cache directory is unavailable: {cache_dir}")

    cursor = 0
    expected_bytes = 0
    observed_bytes = 0
    names: set[str] = set()
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"{label} shard {index} is not an object")
        name = str(shard.get("name", ""))
        start = int(shard.get("start", -1))
        stop = int(shard.get("stop", -1))
        size = int(shard.get("bytes", -1))
        if not name or name in names or start != cursor or stop <= start or size < 0:
            raise ValueError(f"{label} shard ordering is invalid at index {index}")
        names.add(name)
        cursor = stop
        shard_path = cache_dir / name
        if not shard_path.is_file():
            raise FileNotFoundError(f"{label} shard is missing: {shard_path}")
        observed = shard_path.stat().st_size
        if observed != size:
            raise ValueError(
                f"{label} shard size differs for {name}: expected {size}, observed {observed}"
            )
        expected_bytes += size
        observed_bytes += observed
    if cursor != expected_rows:
        raise ValueError(f"{label} shard coverage ends at {cursor}, expected {expected_rows}")
    return {
        "status": "AVAILABLE_SIZE_AND_ORDINALS_VERIFIED",
        "rows": expected_rows,
        "shards": len(shards),
        "bytes": observed_bytes,
        "all_declared_files_present": True,
        "manifest_byte_total_matches": observed_bytes == expected_bytes,
        "payload_sha256_recomputed": False,
        "payload_sha256_policy": (
            "Use the historical complete admission; rehash only by an explicit costly audit."
        ),
        "locator": f"external-cache://{label.lower().replace(' ', '-')}",
    }


def _load_roles(
    workspace: Path, run13: Path
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    roles: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for role in _ROLES:
        manifest_path = run13 / f"{role}_MANIFEST.json"
        manifest = _read_json(manifest_path)
        if manifest.get("role") != role or manifest.get("status") != "COMPLETE":
            raise ValueError(f"Role {role} manifest identity/status is invalid")
        rows_path = Path(str(manifest.get("rows_path", "")))
        if _digest(rows_path) != str(manifest.get("rows_sha256")):
            raise ValueError(f"Role {role} rows hash differs from its manifest")
        frame = pd.read_parquet(rows_path, columns=["sample_token", "group_id"])
        if frame["sample_token"].duplicated().any():
            raise ValueError(f"Role {role} contains duplicate sample tokens")
        groups = {str(value) for value in frame["group_id"].unique()}
        if groups != {str(value) for value in manifest.get("group_ids", [])}:
            raise ValueError(f"Role {role} group list differs from its rows")
        if groups & seen_groups:
            raise ValueError(f"Role {role} overlaps another role by acquisition group")
        seen_groups |= groups
        if len(frame) != int(manifest.get("population_size", -1)):
            raise ValueError(f"Role {role} population differs from its manifest")
        if str(manifest.get("source_sha256", "")) == "":
            raise ValueError(f"Role {role} does not bind the native TRAIN40 source")
        frame = frame.assign(role=role)
        frames.append(frame)
        roles[role] = {
            "queries": len(frame),
            "groups": len(groups),
            "group_ids": sorted(groups),
            "rows_sha256": str(manifest["rows_sha256"]),
            "rows_locator": f"external-metadata://{role}_ROWS.parquet",
        }
        sources.append(_source(workspace, manifest_path))
    return roles, pd.concat(frames, ignore_index=True), sources


def _event_producer(
    workspace: Path,
    fit_dir: Path,
    fit_name: str,
    *,
    input_manifest_sha256: str,
    teacher_manifest_sha256: str,
    expected_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = fit_dir / "CHECKPOINT_RECEIPT.json"
    journal_path = fit_dir / "UPDATE_JOURNAL.json"
    progress_path = fit_dir / "PROGRESS.json"
    checkpoint_path = fit_dir / "checkpoint_last.pt"
    epoch_path = fit_dir / "epoch_18.json"
    receipt = _read_json(receipt_path)
    journal = _read_json(journal_path)
    progress = _read_json(progress_path)
    epoch = _read_json(epoch_path)
    contract = journal.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{fit_name} journal has no contract")
    expected_updates = int(contract.get("updates_limit", -1))
    if (
        receipt.get("status") != "COMPLETE"
        or progress.get("status") != "COMPLETE"
        or int(receipt.get("committed_updates", -1)) != expected_updates
        or int(progress.get("durable_updates", -1)) != expected_updates
        or int(journal.get("durable_updates", -1)) != expected_updates
        or int(epoch.get("updates", -1)) != expected_updates
    ):
        raise ValueError(f"{fit_name} is not a consistent complete endpoint")
    if int(epoch.get("examples", -1)) != expected_rows:
        raise ValueError(f"{fit_name} epoch endpoint is not TRAIN40")
    if contract.get("split_version") != "public_TRAIN40_all_train":
        raise ValueError(f"{fit_name} ancestry is not the all-TRAIN40 split")
    if contract.get("dataset_manifest_sha256") != input_manifest_sha256:
        raise ValueError(f"{fit_name} does not bind the admitted input manifest")
    if contract.get("teacher_manifest_sha256") != teacher_manifest_sha256:
        raise ValueError(f"{fit_name} does not bind the admitted teacher manifest")
    checkpoint_sha256 = _digest(checkpoint_path)
    if checkpoint_sha256 != receipt.get("sha256"):
        raise ValueError(f"{fit_name} checkpoint differs from its complete receipt")
    model = contract.get("model_config", {})
    training = contract.get("training_config", {})
    if model.get("modality") != "event" or int(training.get("epochs", -1)) != 18:
        raise ValueError(f"{fit_name} is not the expected 18-epoch event producer")
    record = {
        "fit": fit_name,
        "status": "COMPLETE_HASH_VERIFIED",
        "checkpoint": _source(workspace, checkpoint_path),
        "checkpoint_receipt_sha256": _digest(receipt_path),
        "scientific_updates": expected_updates,
        "epochs": int(training["epochs"]),
        "seed": int(contract.get("seed", -1)),
        "transport_mode": str(model.get("transport_mode")),
        "protocol_sha256": str(contract.get("protocol_sha256")),
        "split_version": str(contract["split_version"]),
        "input_manifest_sha256": str(contract["dataset_manifest_sha256"]),
        "teacher_manifest_sha256": str(contract["teacher_manifest_sha256"]),
        "fresh_initialization": training.get("initialization_checkpoint") is None,
        "reuse_scope": (
            "Eligible as a final all-TRAIN event producer parent; forbidden for OOF fold "
            "outputs whose held-out groups it saw."
        ),
    }
    return record, [
        _source(workspace, receipt_path),
        _source(workspace, journal_path),
        _source(workspace, progress_path),
        _source(workspace, epoch_path),
    ]


def build_preflight(workspace: Path, output: Path | None = None) -> dict[str, Any]:
    """Validate TRAIN40 identities, caches, and reusable event-producer ancestry.

    ``workspace`` is the directory containing the V12 and V13 sibling
    repositories.  When supplied, ``output`` is a JSON file written only after
    every required check succeeds.
    """
    workspace = workspace.resolve()
    run12 = workspace / _V12_REPO / _RUN12
    run13 = workspace / _V13_REPO / _RUN13
    if not run12.is_dir() or not run13.is_dir():
        raise FileNotFoundError("Required V12/V13 historical run directories are unavailable")

    roles, role_rows, sources = _load_roles(workspace, run13)
    native_path = run12 / "TRAIN40_ROWS.parquet"
    native_sha256 = _digest(native_path)
    native = pd.read_parquet(native_path, columns=["sample_token", "sequence_id"])
    if native["sample_token"].duplicated().any():
        raise ValueError("Native TRAIN40 contains duplicate sample tokens")
    if role_rows["sample_token"].duplicated().any():
        raise ValueError("P/H/V sample tokens overlap")
    if set(native["sample_token"]) != set(role_rows["sample_token"]):
        raise ValueError("P/H/V union differs from native TRAIN40")
    joined = role_rows.merge(native, on="sample_token", validate="one_to_one")
    if not (joined["group_id"].astype(str) == joined["sequence_id"].astype(str)).all():
        raise ValueError("Role group identity differs from native sequence identity")
    if native["sequence_id"].nunique() != 40 or len(native) != 88744:
        raise ValueError("Native population is not the admitted 88,744-query/40-group TRAIN40")
    for role in _ROLES:
        manifest = _read_json(run13 / f"{role}_MANIFEST.json")
        if manifest.get("source_sha256") != native_sha256:
            raise ValueError(f"Role {role} does not bind the current native TRAIN40 rows")
    sources.append(_source(workspace, native_path))

    input_manifest_path = run12 / "INPUT_MANIFEST.json"
    teacher_manifest_path = run12 / "TEACHER_MANIFEST.json"
    teacher_freeze_path = run12 / "TEACHER_FREEZE.json"
    teacher_admission_path = run12 / "TEACHER_COMPLETE_ADMISSION.json"
    input_manifest = _read_json(input_manifest_path)
    teacher_manifest = _read_json(teacher_manifest_path)
    teacher_freeze = _read_json(teacher_freeze_path)
    teacher_admission = _read_json(teacher_admission_path)
    input_manifest_sha256 = _digest(input_manifest_path)
    teacher_manifest_sha256 = _digest(teacher_manifest_path)
    teacher_freeze_sha256 = _digest(teacher_freeze_path)
    for label, manifest in (("producer input", input_manifest), ("DINO teacher", teacher_manifest)):
        if (
            manifest.get("rows_sha256") != native_sha256
            or int(manifest.get("sequence_count", -1)) != 40
            or manifest.get("index_sha256") != teacher_freeze.get("new_TRAIN40_metadata_sha256")
        ):
            raise ValueError(f"{label} manifest does not bind native TRAIN40 metadata")
    if (
        teacher_manifest.get("freeze_sha256") != teacher_freeze_sha256
        or teacher_admission.get("status") != "PASSED"
        or int(teacher_admission.get("complete_rows", -1)) != len(native)
        or int(teacher_admission.get("sequences", -1)) != 40
        or teacher_admission.get("teacher_freeze_sha256") != teacher_freeze_sha256
        or not teacher_admission.get("all_ordinals_and_tokens_exactly_match")
        or not teacher_admission.get("all_targets_finite")
        or teacher_admission.get("TTC_targets_read") is not False
    ):
        raise ValueError("Historical DINO teacher admission is inconsistent")
    input_cache = _validate_shard_cache(
        input_manifest, label="producer input", expected_rows=len(native)
    )
    teacher_cache = _validate_shard_cache(
        teacher_manifest, label="DINO teacher", expected_rows=len(native)
    )
    if teacher_cache["shards"] != int(
        teacher_admission.get("shards_verified", -1)
    ) or teacher_cache["bytes"] != int(teacher_admission.get("verified_bytes", -1)):
        raise ValueError("Current teacher cache size/coverage differs from complete admission")
    sources.extend(
        _source(workspace, path)
        for path in (
            input_manifest_path,
            teacher_manifest_path,
            teacher_freeze_path,
            teacher_admission_path,
        )
    )

    producers: list[dict[str, Any]] = []
    for fit_name in _EVENT_FITS:
        producer, producer_sources = _event_producer(
            workspace,
            run12 / "fits" / fit_name,
            fit_name,
            input_manifest_sha256=input_manifest_sha256,
            teacher_manifest_sha256=teacher_manifest_sha256,
            expected_rows=len(native),
        )
        producers.append(producer)
        sources.extend(producer_sources)
    protocols = {producer["protocol_sha256"] for producer in producers}
    if len(protocols) != 1:
        raise ValueError("A5 and C2F event producers do not share one scientific protocol")

    eap_roots = {
        str(_read_json(run13 / f"{role}_MANIFEST.json").get("eap_root", "")) for role in _ROLES
    }
    if len(eap_roots) != 1:
        raise ValueError("Role manifests disagree on the eAP media root")
    eap_root = Path(next(iter(eap_roots)))
    if not eap_root.is_dir():
        raise FileNotFoundError(f"Declared eAP media root is unavailable: {eap_root}")

    result: dict[str, Any] = {
        "schema": "final_refit_train40_preflight_v1",
        "status": "PASS",
        "scope": "CPU read-only metadata, cache availability, and historical event-parent ancestry",
        "training_started": False,
        "optimizer_updates": 0,
        "test_ground_truth_read": False,
        "population": {
            "queries": len(native),
            "groups": int(native["sequence_id"].nunique()),
            "native_rows_sha256": native_sha256,
            "roles": roles,
            "role_query_total": sum(int(roles[role]["queries"]) for role in _ROLES),
            "role_group_total": sum(int(roles[role]["groups"]) for role in _ROLES),
            "sample_tokens_disjoint": True,
            "acquisition_groups_disjoint": True,
            "union_equals_native_train40": True,
            "group_equals_sequence_identity": True,
        },
        "availability": {
            "role_metadata": "AVAILABLE_AND_SHA256_VERIFIED",
            "eap_media_root": "AVAILABLE_NOT_ENUMERATED",
            "eap_media_locator": "external-data://eAP",
            "producer_input_cache": input_cache,
            "dino_teacher_cache": teacher_cache,
        },
        "teacher": {
            "status": "HISTORICAL_COMPLETE_ADMISSION_BOUND_CURRENT_SIZE_AND_ORDINALS_VERIFIED",
            "manifest_sha256": teacher_manifest_sha256,
            "freeze_sha256": teacher_freeze_sha256,
            "weights_sha256": teacher_admission.get("ordered_teacher_weights_sha256"),
            "model_revision": teacher_freeze.get("model_revision"),
            "training_privilege_only": teacher_freeze.get(
                "teacher_is_training_privilege_not_model_input"
            ),
            "historical_admission_targets_finite": True,
            "current_payload_finiteness_rechecked": False,
            "ttc_targets_read": False,
        },
        "historical_event_producers": producers,
        "parent_ancestry": {
            "all_train_parents_complete": True,
            "shared_protocol_sha256": next(iter(protocols)),
            "final_chain_reuse": "ADMISSIBLE_BY_DATA_ANCESTRY",
            "oof_fold_reuse": "FORBIDDEN_DUE_TO_HELDOUT_GROUP_EXPOSURE",
            "rgb_all_train_parent": "NOT_COVERED_BY_THIS_PREFLIGHT",
            "downstream_compatibility": (
                "Must be checked by the final-refit model adapter before training."
            ),
        },
        "large_payload_policy": {
            "cache_payload_sha256_recomputed": False,
            "reason": (
                "The input cache is about 44 GB and the teacher cache about 1.54 GB; "
                "this preflight verifies all declared paths, sizes, and ordinal coverage "
                "and binds historical admissions."
            ),
            "checkpoint_sha256_recomputed": True,
        },
        "sources": sorted(sources, key=lambda item: str(item["path"])),
    }
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result
