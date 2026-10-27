"""Closed-world archive and local admission for complete cloud RGB returns.

This module deliberately uses only the Python standard library.  It does not
start, stop, or inspect model code and it never deletes remote or local data.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

ROTATIONS = (2, 3, 4)
INDIVIDUAL_RETURN_ROTATIONS = (4,)
FITS = ("R_A5", "R_C2F", "PAIR_R", "R_CTX")
PILOT_SHA256 = "a3dadc7e37e35735c9dad2d70bae5d3e28634a2dd6b802fe10d57ae3f2821cf0"
ARCHIVE_SCHEMA = "ejepa_cloud_rgb_complete_archive_v1"
RESULT_SCHEMA = "ejepa_cloud_rgb_downstream_queue_v1"
MIN_ARCHIVE_RESERVE_BYTES = 2 * 1024**3
RGB_CONTEXT_V5_SHA256 = "cb349a6d514245414f508800b878f9eb3cae44f167ce6c5ec8076696d299f357"
RGB_CONTEXT_V6_SHA256 = "88bf04a2ad2b2938b93b1a1a2a8099de07365992ea0a1ff75ef6bd81684d19ca"


def sha256(path: Path) -> str:
    """Return the SHA-256 of one regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object without changing any embedded path strings."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_by_sha256(
    operational_sources: Sequence[Path], expected: object, *, label: str
) -> Path:
    """Resolve an immutable operational record by bytes, not a future filename."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} lacks a lowercase SHA-256")
    matches = [
        path.resolve(strict=True)
        for path in operational_sources
        if sha256(path) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must resolve to exactly one operational source")
    return matches[0]


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe relative archive path: {value}")
    if any(part in {"", "."} for part in relative.parts):
        raise ValueError(f"Non-canonical archive path: {value}")
    return relative


def _assert_plain_file(path: Path, root: Path) -> Path:
    """Reject symlinks in every component and containment escapes."""
    root = root.resolve(strict=True)
    candidate = root.joinpath(*path.parts) if not path.is_absolute() else path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"File is outside its declared root: {candidate}") from error
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Symlink is forbidden in archive input: {cursor}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Archive input is not a contained regular file: {candidate}")
    return resolved


def _regular_tree(root: Path) -> list[Path]:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Archive tree is not a plain directory: {root}")
    files: list[Path] = []
    for directory, directories, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError(f"Unknown symlink in archive tree: {base / name}")
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Special archive input is forbidden: {path}")
            files.append(path.resolve(strict=True))
    return sorted(files)


def _durable_endpoint(directory: Path, fit: str, rotation: int) -> dict[str, Any]:
    """Fail closed unless a completed endpoint transaction is fully durable."""
    required = (
        "CHECKPOINT_RECEIPT.json",
        "UPDATE_JOURNAL.json",
        "ENDPOINT_ADMISSION.json",
        "STRUCTURAL_LINEAGE.json",
        "checkpoint_last.pt",
    )
    if any(not (directory / name).is_file() for name in required):
        raise FileNotFoundError(f"Endpoint is not sealed: R{rotation}/{fit}")
    receipt = read_json(directory / "CHECKPOINT_RECEIPT.json")
    journal = read_json(directory / "UPDATE_JOURNAL.json")
    admission = read_json(directory / "ENDPOINT_ADMISSION.json")
    completed = receipt.get("completed_updates")
    checkpoint = directory / "checkpoint_last.pt"
    if (
        receipt.get("status") != "COMPLETE"
        or not isinstance(completed, int)
        or completed < 0
        or journal.get("completed_updates") != completed
        or journal.get("durable_updates") != completed
        or journal.get("pending_update_upper") != 0
        or receipt.get("checkpoint_sha256") != sha256(checkpoint)
        or admission.get("rotation") != rotation
        or admission.get("fit_id") != fit
        or admission.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
    ):
        raise ValueError(f"Endpoint transaction is incomplete or inconsistent: R{rotation}/{fit}")
    pointer_path = directory / "CHECKPOINT_POINTER.json"
    if pointer_path.is_file():
        pointer = read_json(pointer_path)
        if (
            pointer.get("completed_updates") != completed
            or pointer.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
            or pointer.get("identity_sha256") != receipt.get("identity_sha256")
        ):
            raise ValueError(f"Endpoint pointer is in flight: R{rotation}/{fit}")
    return {
        "completed_updates": completed,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "admission_sha256": sha256(directory / "ENDPOINT_ADMISSION.json"),
    }


def _writer_fds_linux(roots: Sequence[Path]) -> list[dict[str, object]]:
    """List Linux processes holding a writable descriptor below an allowed root."""
    proc = Path("/proc")
    if os.name != "posix" or not proc.is_dir():
        raise RuntimeError("The remote READY writer check requires Linux /proc")
    resolved_roots = tuple(root.resolve(strict=True) for root in roots)
    writers: list[dict[str, object]] = []
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                flags_line = next(
                    line
                    for line in (process / "fdinfo" / descriptor.name).read_text().splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
                if flags & os.O_ACCMODE == os.O_RDONLY:
                    continue
                target = Path(os.readlink(descriptor))
                if not target.is_absolute():
                    continue
                target = Path(str(target).removesuffix(" (deleted)"))
                if any(target == root or target.is_relative_to(root) for root in resolved_roots):
                    writers.append({"pid": int(process.name), "fd": int(descriptor.name)})
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                StopIteration,
                ValueError,
            ):
                continue
    return writers


def _add_record(
    records: dict[str, dict[str, Any]], archive_path: str, source: Path, source_root: Path
) -> None:
    relative = _safe_relative(archive_path).as_posix()
    if relative == "MANIFEST.json" or relative in records:
        raise ValueError(f"Duplicate/reserved archive path: {relative}")
    source = _assert_plain_file(source, source_root)
    records[relative] = {
        "source_path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def _add_inline_record(
    records: dict[str, dict[str, Any]], archive_path: str, payload: bytes
) -> None:
    """Add archive metadata generated by this exporter, never scientific payload bytes."""
    relative = _safe_relative(archive_path).as_posix()
    if relative == "MANIFEST.json" or relative in records:
        raise ValueError(f"Duplicate/reserved archive path: {relative}")
    records[relative] = {
        "source_path": f"/generated/rgb-complete-return/{relative}",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated_archive_metadata": True,
        "inline_utf8": payload.decode("utf-8"),
    }


def _nested_manifest(
    records: dict[str, dict[str, Any]],
    *,
    prefix: str,
    rotation: int,
    fit: str,
    endpoint: Mapping[str, Any],
) -> None:
    marker = prefix.rstrip("/") + "/"
    files = {
        relative.removeprefix(marker): {
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
        }
        for relative, identity in records.items()
        if relative.startswith(marker)
    }
    manifest = {
        "schema": "ejepa_cloud_checkpoint_snapshot_v1",
        "snapshot": f"complete_{endpoint['checkpoint_sha256'][:12]}",
        "rotation": rotation,
        "fit": fit,
        "completed_updates": endpoint["completed_updates"],
        "checkpoint_sha256": endpoint["checkpoint_sha256"],
        "checkpoint_status": "COMPLETE",
        "files": files,
        "observed_journal_updates": endpoint["completed_updates"],
        "observed_journal_pending_update_upper": 0,
        "live_journal_captured_not_reconstructed": True,
        "training_paused_for_snapshot": False,
        "complete_rgb_return_generated_manifest": True,
    }
    _add_inline_record(
        records,
        f"{prefix}/MANIFEST.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )


def _add_tree(
    records: dict[str, dict[str, Any]],
    prefix: str,
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> None:
    resolved = root.resolve(strict=True)
    for path in _regular_tree(resolved):
        if path.name in excluded_names:
            continue
        _add_record(records, f"{prefix}/{path.relative_to(resolved).as_posix()}", path, resolved)


def _validate_result(run: Path, rotation: int) -> tuple[dict[str, Any], list[Path]]:
    path = run / "RGB_DOWNSTREAM_RESULT.json"
    result = read_json(path)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("stage") != "RGB_OOF"
        or result.get("status") != "COMPLETE"
        or result.get("rotation") != rotation
        or result.get("event_branch_executed") is not False
        or result.get("event_inputs_read") is not False
        or result.get("emitted_interface") != "PHASE17_RAW_RGB"
    ):
        raise ValueError(f"R{rotation} is not a complete RGB_OOF result")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"R{rotation} result has no artifact closure")
    paths: list[Path] = []
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"R{rotation} has a malformed artifact record")
        relative = _safe_relative(item["path"])
        name = relative.as_posix()
        if name in seen:
            raise ValueError(f"R{rotation} repeats artifact {name}")
        seen.add(name)
        candidate = _assert_plain_file(Path(*relative.parts), run)
        if candidate.stat().st_size != item.get("bytes") or sha256(candidate) != item.get("sha256"):
            raise ValueError(f"R{rotation} artifact differs from RESULT: {name}")
        paths.append(candidate)
    required_suffixes = {
        f"rotations/{rotation}/FOLD_TRAINING_UNIT.json",
        f"rotations/{rotation}/caches/PAIR_R.npz",
        f"rotations/{rotation}/caches/PAIR_R.json",
        f"rotations/{rotation}/caches/R_CTX.npz",
        f"rotations/{rotation}/caches/R_CTX.json",
        f"rotations/{rotation}/normalizers/R_CTX.json",
        f"rotations/{rotation}/generation/R_CTX/fragment.npz",
        f"rotations/{rotation}/generation/R_CTX/fragment.json",
    }
    if not required_suffixes.issubset(seen):
        missing = sorted(required_suffixes - seen)
        raise ValueError(f"R{rotation} result artifact closure is missing: {missing}")
    return result, paths


def _validate_rgb_path_compatibility(
    downstream: Path, operational_by_name: Mapping[str, Path]
) -> Path:
    """Bind the native-Path adapter sidecar to migration and explicit source bytes."""
    sidecar = downstream / "RGB_NATIVE_PATH_COMPATIBILITY.json"
    value = read_json(sidecar)
    overlay = operational_by_name.get("runpod_rgb_path_compat_v1.py")
    bootstrap = operational_by_name.get("runpod_rgb_bootstrap_v1.py")
    if (
        overlay is None
        or bootstrap is None
        or value.get("schema") != "ejepa_rgb_native_path_compatibility_v1"
        or value.get("status") != "PASS"
        or value.get("overlay_sha256") != sha256(overlay)
        or value.get("original_bootstrap_sha256") != sha256(bootstrap)
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("native_path_isinstance_preserved") is not True
        or value.get("translation_constructor_unchanged") is not True
        or value.get("scientific_sources_rewritten") is not False
        or value.get("checkpoint_bytes_rewritten") is not False
    ):
        raise ValueError("RGB native Path compatibility sidecar/source binding changed")
    return sidecar


def _validate_rgb_operational_recovery_v2(
    downstream: Path, operational_by_name: Mapping[str, Path]
) -> Path:
    """Bind the bounded ENOENT retry/heartbeat wrapper to its two exact dependencies."""
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V2.json"
    value = read_json(sidecar)
    recovery = operational_by_name.get("runpod_rgb_recovery_v2.py")
    path_compat = operational_by_name.get("runpod_rgb_path_compat_v1.py")
    quota = operational_by_name.get("runpod_quota_race_overlay_v1.py")
    dependencies = value.get("dependencies")
    dependency_by_name = (
        {Path(path).name: sha for path, sha in dependencies.items()}
        if isinstance(dependencies, dict)
        and all(
            isinstance(path, str) and isinstance(sha, str)
            for path, sha in dependencies.items()
        )
        else {}
    )
    if (
        recovery is None
        or path_compat is None
        or quota is None
        or value.get("schema") != "ejepa_rgb_operational_recovery_v2"
        or value.get("status") != "PASS"
        or value.get("source_sha256") != sha256(recovery)
        or dependency_by_name
        != {
            path_compat.name: sha256(path_compat),
            quota.name: sha256(quota),
        }
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("heartbeat_emitted_by_active_worker") is not True
        or value.get("heartbeat_interval_seconds") != 30
        or value.get("quota_retry_scope") != "GNU-du status1 ENOENT only"
        or value.get("scientific_sources_rewritten") is not False
    ):
        raise ValueError("RGB operational recovery v2 sidecar/source binding changed")
    return sidecar


def _validate_optional_rgb_operational_recovery_v3(
    downstream: Path, operational_by_name: Mapping[str, Path]
) -> Path | None:
    """Validate V3 only when a worker actually restarted through that wrapper."""
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V3.json"
    if not sidecar.is_file():
        return None
    value = read_json(sidecar)
    recovery_v3 = operational_by_name.get("runpod_rgb_recovery_v3.py")
    recovery_v2 = operational_by_name.get("runpod_rgb_recovery_v2.py")
    if (
        recovery_v3 is None
        or recovery_v2 is None
        or value.get("schema") != "ejepa_rgb_operational_recovery_v3"
        or value.get("status") != "PASS"
        or value.get("source_sha256") != sha256(recovery_v3)
        or value.get("frozen_v2_sha256") != sha256(recovery_v2)
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("heartbeat_receipt_precedes_optional_external_guard") is not True
        or value.get("external_guard_success_inferred") is not False
        or value.get("optimizer_updates_inferred") is not False
        or value.get("frozen_v2_main_delegated") is not True
        or value.get("scientific_sources_rewritten") is not False
    ):
        raise ValueError("RGB operational recovery v3 sidecar/source binding changed")
    return sidecar


def _validate_optional_rgb_history_v4(
    downstream: Path, operational_sources: Sequence[Path]
) -> Path | None:
    """Preserve a historical V4 adapter without treating it as the active chain."""
    sidecar = downstream / "RGB_HISTORY_METADATA_ADAPTER_V4.json"
    if not sidecar.is_file():
        return None
    value = read_json(sidecar)
    source = _source_by_sha256(
        operational_sources, value.get("source_sha256"), label="historical RGB V4"
    )
    if (
        source.name != "runpod_rgb_metadata_v4.py"
        or value.get("schema") != "ejepa_rgb_history_metadata_adapter_v4"
        or value.get("status") != "PASS"
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("scientific_sources_rewritten") is not False
    ):
        raise ValueError("Historical RGB V4 sidecar/source binding changed")
    return sidecar


def _validate_rgb_operational_recovery_v5(
    downstream: Path, operational_sources: Sequence[Path]
) -> tuple[Path, dict[str, Path]]:
    """Require the active, audit-bound three-field metadata comparison repair."""
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V5.json"
    value = read_json(sidecar)
    bindings = {
        "source": _source_by_sha256(
            operational_sources, value.get("source_sha256"), label="RGB context V5"
        ),
        "v3": _source_by_sha256(
            operational_sources, value.get("recovery_v3_sha256"), label="RGB recovery V3"
        ),
        "repair": _source_by_sha256(
            operational_sources,
            value.get("metadata_repair_sha256"),
            label="RGB metadata repair",
        ),
        "audit": _source_by_sha256(
            operational_sources,
            value.get("metadata_stability_audit_sha256"),
            label="RGB metadata stability audit",
        ),
    }
    if (
        sha256(bindings["source"]) != RGB_CONTEXT_V5_SHA256
        or bindings["source"].name != "runpod_rgb_context_v5.py"
        or bindings["repair"].name != "oof_rgb_metadata_fix.py"
        or value.get("schema") != "ejepa_rgb_context_metadata_recovery_v5"
        or value.get("status") != "PASS"
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("excluded_query_reference_fields")
        != ["query_age_us", "query_time_us", "sample_token"]
        or value.get("materializer_ast_comparisons_replaced") != 1
        or value.get("materializer_and_imported_alias_scoped_to_queue_call") is not True
        or value.get("rgb_pixels_changed") is not False
        or value.get("scientific_recipes_changed") is not False
        or value.get("scientific_inputs_changed") is not False
        or value.get("scientific_sources_rewritten") is not False
    ):
        raise ValueError("RGB operational recovery V5 sidecar/source binding changed")
    audit = read_json(bindings["audit"])
    invariants = audit.get("invariants")
    if (
        audit.get("schema") != "ejepa_rgb_observation_metadata_stability_audit_v1"
        or audit.get("status") != "PASS"
        or not isinstance(invariants, dict)
        or invariants.get("all_observed_duplicate_differences_are_exact_three_query_fields")
        is not True
        or invariants.get("sensor_roi_group_role_conflicts") != 0
        or invariants.get("model_forward_calls") != 0
        or invariants.get("remote_updates") != 0
    ):
        raise ValueError("RGB metadata stability audit contract changed")
    return sidecar, bindings


def _media_config_source(
    operational_sources: Sequence[Path], expected_authority: object
) -> tuple[Path, dict[str, Any]]:
    """Find the current append-only config by its immutable header authority."""
    matches: list[tuple[Path, dict[str, Any]]] = []
    for raw in operational_sources:
        path = raw.resolve(strict=True)
        try:
            value = read_json(path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if value.get("schema") != "ejepa_rgb_media_scope_config_v1":
            continue
        header = {key: item for key, item in value.items() if key != "scopes"}
        if _canonical_sha256(header) == expected_authority:
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError("RGB media authority must resolve to one append-only config")
    return matches[0]


def _validate_rgb_operational_recovery_v6(
    downstream: Path,
    rotation: int,
    operational_sources: Sequence[Path],
) -> tuple[Path, Path, list[tuple[str, Path, Path, list[Path]]], Path]:
    """Validate V6 and return exact scope plan/manifest/member-hash evidence."""
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V6.json"
    value = read_json(sidecar)
    source = _source_by_sha256(
        operational_sources, value.get("source_sha256"), label="RGB context V6"
    )
    v5 = _source_by_sha256(
        operational_sources, value.get("context_v5_sha256"), label="RGB context V5"
    )
    if (
        sha256(source) != RGB_CONTEXT_V6_SHA256
        or source.name != "runpod_rgb_context_v6.py"
        or sha256(v5) != RGB_CONTEXT_V5_SHA256
        or value.get("schema") != "ejepa_rgb_frame_gate_routing_v6"
        or value.get("status") != "PASS"
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("reader_routing_scope") != "materializer_call_only"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("initial_media_config_sha256"))
        )
        or value.get("original_frames_crops_clocks_changed") is not False
        or value.get("model_inference_requires_scope_admission") is not True
    ):
        raise ValueError("RGB operational recovery V6 sidecar/source binding changed")
    config_path, config = _media_config_source(
        operational_sources, value.get("media_authority_sha256")
    )
    scopes = config.get("scopes")
    if not isinstance(scopes, list):
        raise ValueError("RGB media config scopes are malformed")
    scope_map = {
        (item.get("rotation"), item.get("role")): item
        for item in scopes
        if isinstance(item, dict)
    }
    results: list[tuple[str, Path, Path, list[Path]]] = []
    if (downstream / "RGB_OPERATIONAL_RECOVERY_V8.json").is_file():
        return sidecar, config_path, results, source
    media_root = Path(str(config.get("media_root"))).resolve(strict=True)
    for role in ("refiner", "generation"):
        scope = scope_map.get((rotation, role))
        if not isinstance(scope, dict):
            raise ValueError(f"R{rotation}/{role} media scope is absent")
        admission_path = downstream / f"RGB_MEDIA_SCOPE_ADMISSION_R{rotation}_{role}.json"
        admission = read_json(admission_path)
        plan = Path(str(scope.get("plan_path"))).resolve(strict=True)
        manifest = Path(str(scope.get("manifest_path"))).resolve(strict=True)
        if (
            admission.get("schema") != "ejepa_rgb_media_scope_admission_v1"
            or admission.get("status") != "PASS"
            or admission.get("rotation") != rotation
            or admission.get("role") != role
            or admission.get("scope_evidence_sha256") != _canonical_sha256(scope)
            or admission.get("plan_sha256") != sha256(plan)
            or admission.get("manifest_sha256") != sha256(manifest)
            or admission.get("authority_sha256") != value.get("media_authority_sha256")
            or admission.get("media_root") != str(config.get("media_root"))
            or admission.get("model_inference_before_admission") is not False
            or admission.get("optimizer_updates_inferred") is not False
        ):
            raise ValueError(f"R{rotation}/{role} media scope admission changed")
        plan_value, manifest_value = read_json(plan), read_json(manifest)
        plan_payload = {key: item for key, item in plan_value.items() if key != "receipt_sha256"}
        manifest_payload = {
            key: item for key, item in manifest_value.items() if key != "receipt_sha256"
        }
        if (
            plan_value.get("schema") != "ejepa_rgb_frame_supplement_plan_v1"
            or plan_value.get("status") != "PLANNED_HEADERS_VERIFIED"
            or plan_value.get("rotations") != [rotation]
            or plan_value.get("roles") != [role]
            or plan_value.get("receipt_sha256") != _canonical_sha256(plan_payload)
            or manifest_value.get("schema") != "ejepa_rgb_frame_supplement_v1"
            or manifest_value.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
            or manifest_value.get("plan_sha256") != sha256(plan)
            or manifest_value.get("receipt_sha256") != _canonical_sha256(manifest_payload)
            or manifest_value.get("payload_recompressed") is not False
            or manifest_value.get("decoded_pixel_parity") is not True
        ):
            raise ValueError(f"R{rotation}/{role} media plan/manifest changed")
        receipt_paths: list[Path] = []
        for record in manifest_value.get("tars", []):
            if not isinstance(record, dict):
                raise ValueError("RGB sparse TAR receipt record is malformed")
            relative = _safe_relative(str(record.get("relative")))
            tar_path = media_root.joinpath(*relative.parts)
            receipt_path = tar_path.with_suffix(tar_path.suffix + ".json").resolve(strict=True)
            if read_json(receipt_path) != record:
                raise ValueError("RGB sparse TAR member-hash receipt changed")
            members = record.get("members")
            record_payload = {
                key: item for key, item in record.items() if key != "receipt_sha256"
            }
            if (
                record.get("schema") != "ejepa_rgb_sparse_tar_receipt_v1"
                or record.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
                or record.get("receipt_sha256") != _canonical_sha256(record_payload)
                or not isinstance(members, list)
                or not members
                or any(
                    not isinstance(member, dict)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(member.get("compressed_payload_sha256"))
                    )
                    for member in members
                )
            ):
                raise ValueError("RGB encoded member hash evidence is incomplete")
            receipt_paths.append(receipt_path)
        results.append((role, plan, manifest, receipt_paths))
    return sidecar, config_path, results, source


def _validate_rgb_operational_recovery_v7(
    downstream: Path, operational_sources: Sequence[Path]
) -> tuple[Path, Path]:
    """Require the final process-local stat barrier without pinning a future hash."""
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V7.json"
    value = read_json(sidecar)
    source = _source_by_sha256(
        operational_sources, value.get("source_sha256"), label="RGB context V7"
    )
    v6 = _source_by_sha256(
        operational_sources, value.get("context_v6_sha256"), label="RGB context V6"
    )
    if (
        source.name != "runpod_rgb_context_v7.py"
        or sha256(v6) != RGB_CONTEXT_V6_SHA256
        or value.get("schema") != "ejepa_rgb_media_stat_cache_v7"
        or value.get("status") != "PASS"
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("barrier_fields")
        != ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"]
        or value.get("barrier_scope") != "plan_manifest_all_tars_all_sidecars"
        or value.get("cache_scope") != "current_process_only"
        or value.get("missing_or_changed_after_admission") != "FAIL_CLOSED"
        or value.get("missing_before_admission") != "NO_CACHE_WAIT"
    ):
        raise ValueError("RGB operational recovery V7 sidecar/source binding changed")
    return sidecar, source


def _validate_rgb_operational_recovery_v8(
    downstream: Path,
    rotation: int,
    operational_sources: Sequence[Path],
    original_media_config: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    """Validate the compositional R4 media genealogy without pinning future V8 bytes."""
    if rotation != 4:
        raise ValueError("Compositional RGB V8 is authorized only for R4 return")
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V8.json"
    value = read_json(sidecar)
    source = _source_by_sha256(
        operational_sources, value.get("source_sha256"), label="RGB context V8"
    )
    v7 = _source_by_sha256(
        operational_sources, value.get("context_v7_sha256"), label="RGB context V7"
    )
    layout = _source_by_sha256(
        operational_sources, value.get("layout_sha256"), label="RGB V8 media layout"
    )
    layout_value = read_json(layout)
    layout_header = {key: item for key, item in layout_value.items() if key != "scopes"}
    if (
        source.name != "runpod_rgb_context_v8.py"
        or v7.name != "runpod_rgb_context_v7.py"
        or value.get("schema") != "ejepa_rgb_compositional_context_v8"
        or value.get("status") != "PASS"
        or value.get("layout_authority_sha256") != _canonical_sha256(layout_header)
        or value.get("original_config_sha256") != sha256(original_media_config)
        or value.get("migration_binding_sha256")
        != sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("allowed_physical_roots")
        != ["/opt/ejepa_rgb_frames", "/workspace/ejepa_rgb_frames"]
        or value.get("physical_sidecar_roles_relabelled") is not False
        or value.get("existing_r3_admissions_changed") is not False
        or value.get("optimizer_updates_inferred") is not False
        or layout_value.get("schema") != "ejepa_rgb_compositional_media_layout_v8"
        or layout_value.get("status") != "PASS"
        or layout_value.get("original_config_sha256") != sha256(original_media_config)
        or Path(str(layout_value.get("original_config_path"))).resolve(strict=True)
        != original_media_config.resolve(strict=True)
        or layout_value.get("allowed_physical_roots")
        != ["/opt/ejepa_rgb_frames", "/workspace/ejepa_rgb_frames"]
    ):
        raise ValueError("RGB operational recovery V8 sidecar/layout binding changed")
    scopes = {
        (item.get("rotation"), item.get("role")): item
        for item in layout_value.get("scopes", [])
        if isinstance(item, dict)
    }
    expected = {
        "refiner": ("DIRECT", 4, "refiner", "/workspace/ejepa_rgb_frames"),
        "generation": ("REUSE", 3, "refiner", "/opt/ejepa_rgb_frames"),
    }
    combined_authority = _canonical_sha256(
        {
            "original_authority_sha256": read_json(
                downstream / "RGB_OPERATIONAL_RECOVERY_V6.json"
            )["media_authority_sha256"],
            "layout_authority_sha256": value["layout_authority_sha256"],
        }
    )
    evidence: list[dict[str, Any]] = []
    for role, identity in expected.items():
        item = scopes.get((4, role))
        if not isinstance(item, dict):
            raise ValueError(f"R4/{role} V8 compositional scope is absent")
        mode, physical_rotation, physical_role, physical_root = identity
        if (
            item.get("mode") != mode
            or item.get("physical_rotation") != physical_rotation
            or item.get("physical_role") != physical_role
            or item.get("physical_root") != physical_root
        ):
            raise ValueError(f"R4/{role} V8 physical genealogy changed")
        paths: dict[str, Path] = {}
        names = ["target_plan", "source_plan", "source_manifest"]
        if mode == "REUSE":
            names.append("reuse_proof")
        for name in names:
            path = Path(str(item.get(f"{name}_path"))).resolve(strict=True)
            if item.get(f"{name}_sha256") != sha256(path):
                raise ValueError(f"R4/{role} V8 {name} bytes changed")
            paths[name] = path
        target_plan = read_json(paths["target_plan"])
        source_plan = read_json(paths["source_plan"])
        manifest = read_json(paths["source_manifest"])
        for label, plan in (("target", target_plan), ("source", source_plan)):
            payload = {key: item for key, item in plan.items() if key != "receipt_sha256"}
            if (
                plan.get("schema") != "ejepa_rgb_frame_supplement_plan_v1"
                or plan.get("status") != "PLANNED_HEADERS_VERIFIED"
                or plan.get("receipt_sha256") != _canonical_sha256(payload)
            ):
                raise ValueError(f"R4/{role} V8 {label} plan receipt changed")
        manifest_payload = {
            key: item for key, item in manifest.items() if key != "receipt_sha256"
        }
        if (
            manifest.get("schema") != "ejepa_rgb_frame_supplement_v1"
            or manifest.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
            or manifest.get("plan_sha256") != sha256(paths["source_plan"])
            or manifest.get("receipt_sha256") != _canonical_sha256(manifest_payload)
            or manifest.get("payload_recompressed") is not False
            or manifest.get("decoded_pixel_parity") is not True
        ):
            raise ValueError(f"R4/{role} V8 physical manifest changed")
        admission_path = downstream / f"RGB_MEDIA_SCOPE_ADMISSION_R4_{role}.json"
        admission = read_json(admission_path)
        physical = admission.get("physical_admission")
        if (
            admission.get("schema") != "ejepa_rgb_compositional_scope_admission_v8"
            or admission.get("status") != "PASS"
            or admission.get("rotation") != 4
            or admission.get("role") != role
            or admission.get("target_plan_sha256") != sha256(paths["target_plan"])
            or admission.get("physical_rotation") != physical_rotation
            or admission.get("physical_role") != physical_role
            or admission.get("physical_root") != physical_root
            or admission.get("composition_mode") != mode
            or admission.get("authority_sha256") != combined_authority
            or admission.get("physical_sidecar_roles_relabelled") is not False
            or admission.get("optimizer_updates_inferred") is not False
            or not isinstance(physical, dict)
            or physical.get("schema") != "ejepa_rgb_media_scope_admission_v1"
            or physical.get("status") != "PASS"
            or physical.get("rotation") != physical_rotation
            or physical.get("role") != physical_role
            or physical.get("plan_sha256") != sha256(paths["source_plan"])
            or physical.get("manifest_sha256") != sha256(paths["source_manifest"])
            or physical.get("media_root") != physical_root
            or physical.get("model_inference_before_admission") is not False
            or physical.get("optimizer_updates_inferred") is not False
        ):
            raise ValueError(f"R4/{role} V8 compositional admission changed")
        if mode == "DIRECT":
            if (
                admission.get("reuse_proof_sha256") is not None
                or admission.get("reuse_proof_receipt_sha256") is not None
                or sha256(paths["target_plan"]) != sha256(paths["source_plan"])
            ):
                raise ValueError("R4/refiner direct scope unexpectedly has reuse proof")
        else:
            proof = read_json(paths["reuse_proof"])
            proof_payload = {key: item for key, item in proof.items() if key != "receipt_sha256"}
            if (
                proof.get("schema") != "ejepa_rgb_media_scope_reuse_proof_v1"
                or proof.get("status") != "PASS"
                or proof.get("receipt_sha256") != _canonical_sha256(proof_payload)
                or admission.get("reuse_proof_sha256") != sha256(paths["reuse_proof"])
                or admission.get("reuse_proof_receipt_sha256") != proof.get("receipt_sha256")
                or proof.get("physical_sidecar_roles_relabelled") is not False
                or proof.get("optimizer_updates") != 0
                or proof.get("same_exact_member_names_and_payload_bytes") is not True
                or proof.get("same_logical_tar_paths") is not True
                or proof.get("target", {}).get("rotation") != 4
                or proof.get("target", {}).get("role") != role
                or proof.get("target", {}).get("plan_sha256")
                != sha256(paths["target_plan"])
                or proof.get("physical", {}).get("rotation") != physical_rotation
                or proof.get("physical", {}).get("role") != physical_role
                or proof.get("physical", {}).get("plan_sha256")
                != sha256(paths["source_plan"])
                or proof.get("physical", {}).get("manifest_sha256")
                != sha256(paths["source_manifest"])
            ):
                raise ValueError("R4/generation reuse proof genealogy changed")
        receipts: list[Path] = []
        for record in manifest.get("tars", []):
            relative = _safe_relative(str(record.get("relative")))
            receipt_path = Path(physical_root).joinpath(*relative.parts).with_suffix(
                Path(relative.name).suffix + ".json"
            ).resolve(strict=True)
            record_payload = {
                key: item for key, item in record.items() if key != "receipt_sha256"
            }
            if (
                read_json(receipt_path) != record
                or record.get("schema") != "ejepa_rgb_sparse_tar_receipt_v1"
                or record.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
                or record.get("receipt_sha256") != _canonical_sha256(record_payload)
                or not isinstance(record.get("members"), list)
                or not record["members"]
            ):
                raise ValueError("R4 V8 physical sparse TAR receipt changed")
            receipts.append(receipt_path)
        evidence.append(
            {
                "role": role,
                "admission": admission_path,
                "paths": paths,
                "physical_root": physical_root,
                "physical_rotation": physical_rotation,
                "physical_role": physical_role,
                "mode": mode,
                "tar_receipts": receipts,
            }
        )
    return sidecar, source, layout, evidence


def collect_rgb_return(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    rotations: Sequence[int] = ROTATIONS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the complete closure and build its immutable per-file manifest."""
    selected = tuple(rotations)
    if selected not in {ROTATIONS, INDIVIDUAL_RETURN_ROTATIONS}:
        raise ValueError("RGB return rotations must be exactly [2,3,4] or individual [4]")
    if (
        set(downstream_runs) != set(selected)
        or set(a5_runs) != set(selected)
        or set(c2f_runs) != set(selected)
    ):
        raise ValueError("RGB run-root maps must equal the selected rotations exactly")
    records: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, Any] = {}
    run_roots: list[Path] = []
    operational_by_name = {
        source.resolve(strict=True).name: source.resolve(strict=True)
        for source in operational_sources
    }
    historical_v4_rotations: list[int] = []
    compositional_v8_rotations: list[int] = []
    media_scope_evidence: dict[str, Any] = {}
    archived_media_config: Path | None = None
    for rotation in selected:
        downstream = downstream_runs[rotation].resolve(strict=True)
        a5 = a5_runs[rotation].resolve(strict=True)
        c2f = c2f_runs[rotation].resolve(strict=True)
        run_roots.extend((downstream, a5, c2f))
        if downstream.name != f"r{rotation}_rgb_downstream_v1":
            raise ValueError(f"Unexpected downstream root for R{rotation}: {downstream}")
        if a5.name != f"r{rotation}_rgb_v3" or c2f.name != f"r{rotation}_c2f_v3":
            raise ValueError(f"Unexpected producer root for R{rotation}")
        result, artifact_paths = _validate_result(downstream, rotation)
        authority = result.get("authority")
        source_bindings = {
            "queue_entrypoint_sha256": "runpod_rgb_queue_v1.py",
            "queue_library_sha256": "rgb_queue.py",
            "queue_endpoint_router_sha256": "rgb_queue_endpoints.py",
            "rgb_downstream_coverage_checker_sha256": "rgb_downstream_coverage.py",
        }
        if not isinstance(authority, dict):
            raise ValueError(f"R{rotation} result lacks its operational authority")
        for field, name in source_bindings.items():
            source = operational_by_name.get(name)
            if source is None or authority.get(field) != sha256(source):
                raise ValueError(f"R{rotation} operational source binding changed: {name}")
        attribution = read_json(downstream / "RGB_HEAD_GIT_ATTRIBUTION.json")
        if attribution.get("rgb_pilot_archive_sha256") != PILOT_SHA256:
            raise ValueError("RGB pilot input binding changed")
        admission = read_json(downstream / "RGB_DOWNSTREAM_ADMISSION.json")
        deployment_binding = downstream / "RGB_DEPLOYMENT_BINDING.json"
        path_compatibility = _validate_rgb_path_compatibility(
            downstream, operational_by_name
        )
        operational_recovery = _validate_rgb_operational_recovery_v2(
            downstream, operational_by_name
        )
        optional_recovery_v3 = _validate_optional_rgb_operational_recovery_v3(
            downstream, operational_by_name
        )
        historical_v4 = _validate_optional_rgb_history_v4(
            downstream, operational_sources
        )
        operational_recovery_v5, _ = _validate_rgb_operational_recovery_v5(
            downstream, operational_sources
        )
        operational_recovery_v6, media_config, scope_evidence, _ = (
            _validate_rgb_operational_recovery_v6(
                downstream, rotation, operational_sources
            )
        )
        operational_recovery_v7, _ = _validate_rgb_operational_recovery_v7(
            downstream, operational_sources
        )
        recovery_v8: Path | None = None
        v8_evidence: list[dict[str, Any]] = []
        if (downstream / "RGB_OPERATIONAL_RECOVERY_V8.json").is_file():
            recovery_v8, _, _, v8_evidence = (
                _validate_rgb_operational_recovery_v8(
                    downstream, rotation, operational_sources, media_config
                )
            )
            compositional_v8_rotations.append(rotation)
        elif rotation == 4:
            raise ValueError("R4 complete return requires compositional recovery V8")
        if (
            admission.get("schema") != "ejepa_cloud_rgb_downstream_admission_v1"
            or admission.get("status") != "PASS"
            or admission.get("task") != {"rotation": rotation, "fit": "RGB_DOWNSTREAM"}
            or admission.get("migration_binding_sha256")
            != sha256(downstream / "MIGRATION_BINDING.json")
            or admission.get("external_deployment_binding_sha256")
            != sha256(deployment_binding)
            or authority.get("rgb_downstream_admission_sha256")
            != sha256(downstream / "RGB_DOWNSTREAM_ADMISSION.json")
        ):
            raise ValueError(f"R{rotation} downstream/deployment admission changed")
        downstream_prefix = f"roots/r{rotation}_downstream"
        for name in (
            "RGB_DOWNSTREAM_ADMISSION.json",
            "RGB_DOWNSTREAM_RESULT.json",
            "RGB_DEPLOYMENT_BINDING.json",
            path_compatibility.name,
            operational_recovery.name,
            operational_recovery_v5.name,
            operational_recovery_v6.name,
            operational_recovery_v7.name,
            "RGB_ENDPOINT_R_A5.json",
            "RGB_ENDPOINT_R_C2F.json",
            "RGB_HEAD_GIT_ATTRIBUTION.json",
            "MIGRATION_BINDING.json",
            "SCIENTIFIC_CONFIG.json",
        ):
            _add_record(records, f"{downstream_prefix}/{name}", downstream / name, downstream)
        if recovery_v8 is not None:
            _add_record(
                records, f"{downstream_prefix}/{recovery_v8.name}", recovery_v8, downstream
            )
        if optional_recovery_v3 is not None:
            _add_record(
                records,
                f"{downstream_prefix}/{optional_recovery_v3.name}",
                optional_recovery_v3,
                downstream,
            )
        if historical_v4 is not None:
            historical_v4_rotations.append(rotation)
            _add_record(
                records,
                f"{downstream_prefix}/{historical_v4.name}",
                historical_v4,
                downstream,
            )
        if archived_media_config is None:
            _add_record(
                records,
                "operations/rgb_media_scope/RGB_MEDIA_SCOPE_CONFIG.json",
                media_config,
                media_config.parent,
            )
            archived_media_config = media_config
        elif media_config != archived_media_config:
            raise ValueError("RGB runs use different append-only media configs")
        for role, plan, media_manifest, tar_receipts in scope_evidence:
            scope_prefix = f"operations/rgb_media_scope/R{rotation}_{role}"
            admission_path = (
                downstream / f"RGB_MEDIA_SCOPE_ADMISSION_R{rotation}_{role}.json"
            )
            _add_record(
                records,
                f"{downstream_prefix}/{admission_path.name}",
                admission_path,
                downstream,
            )
            _add_record(records, f"{scope_prefix}/plan.json", plan, plan.parent)
            _add_record(
                records,
                f"{scope_prefix}/manifest.json",
                media_manifest,
                media_manifest.parent,
            )
            receipt_records: list[dict[str, Any]] = []
            for receipt_path in tar_receipts:
                receipt = read_json(receipt_path)
                relative = _safe_relative(str(receipt["relative"]))
                archive_relative = (
                    f"{scope_prefix}/tar_receipts/{relative.as_posix()}.json"
                )
                _add_record(records, archive_relative, receipt_path, receipt_path.parent)
                receipt_records.append(
                    {
                        "relative": relative.as_posix(),
                        "receipt_sha256": sha256(receipt_path),
                        "member_count": len(receipt["members"]),
                    }
                )
            media_scope_evidence[f"R{rotation}_{role}"] = {
                "plan_sha256": sha256(plan),
                "manifest_sha256": sha256(media_manifest),
                "physical_root": read_json(admission_path)["media_root"],
                "tar_payloads_included": False,
                "tar_receipts": receipt_records,
            }
        for scope in v8_evidence:
            role = str(scope["role"])
            scope_prefix = f"operations/rgb_media_scope/R{rotation}_{role}"
            admission_path = Path(scope["admission"])
            _add_record(
                records,
                f"{downstream_prefix}/{admission_path.name}",
                admission_path,
                downstream,
            )
            archived_paths: dict[str, str] = {}
            for label, path in scope["paths"].items():
                archive_relative = f"{scope_prefix}/{label}/{path.name}"
                _add_record(records, archive_relative, path, path.parent)
                archived_paths[str(label)] = archive_relative
            receipt_records = []
            for receipt_path in scope["tar_receipts"]:
                receipt = read_json(receipt_path)
                relative = _safe_relative(str(receipt["relative"]))
                archive_relative = f"{scope_prefix}/tar_receipts/{relative.as_posix()}.json"
                _add_record(records, archive_relative, receipt_path, receipt_path.parent)
                receipt_records.append(
                    {
                        "relative": relative.as_posix(),
                        "receipt_sha256": sha256(receipt_path),
                        "member_count": len(receipt["members"]),
                    }
                )
            media_scope_evidence[f"R{rotation}_{role}"] = {
                "admission_schema": "ejepa_rgb_compositional_scope_admission_v8",
                "composition_mode": scope["mode"],
                "physical_rotation": scope["physical_rotation"],
                "physical_role": scope["physical_role"],
                "physical_root": scope["physical_root"],
                "archived_genealogy": archived_paths,
                "tar_payloads_included": False,
                "tar_receipts": receipt_records,
            }
        downstream_authorization = authorization_root / f"r{rotation}_rgb_downstream"
        downstream_authority_files = {
            "assignment_sha256": downstream_authorization
            / f"SCIENTIFIC_R{rotation}_RGB_DOWNSTREAM.json",
            "exclusion_sha256": downstream_authorization / "LOCAL_QUEUE_EXCLUSION.json",
            "budget_sha256": downstream_authorization / "USER_BUDGET_24USD.json",
        }
        if not downstream_authority_files["budget_sha256"].is_file():
            downstream_authority_files["budget_sha256"] = (
                authorization_root / "USER_BUDGET_24USD.json"
            )
        for field, source in downstream_authority_files.items():
            if authority.get(field) != sha256(source):
                raise ValueError(f"R{rotation} downstream authority changed: {source.name}")
            _add_record(
                records,
                f"{downstream_prefix}/cloud_authorization/{source.name}",
                source,
                source.parent,
            )
        for fit, producer in (("R_A5", a5), ("R_C2F", c2f)):
            routed = downstream / f"rotations/{rotation}/fits/{fit}"
            expected = producer / f"rotations/{rotation}/fits/{fit}"
            route = read_json(downstream / f"RGB_ENDPOINT_{fit}.json")
            if (
                not routed.is_symlink()
                or routed.resolve(strict=True) != expected.resolve(strict=True)
                or route.get("routing_method") != "DIRECTORY_SYMLINK_NO_CHECKPOINT_COPY"
                or Path(str(route.get("source_endpoint"))).resolve(strict=True)
                != expected.resolve(strict=True)
            ):
                raise ValueError(f"R{rotation}/{fit} is not its explicit producer symlink")
        for source in artifact_paths:
            relative = source.relative_to(downstream).as_posix()
            _add_record(records, f"{downstream_prefix}/{relative}", source, downstream)
        for fit, producer in (("R_A5", a5), ("R_C2F", c2f)):
            directory = producer / f"rotations/{rotation}/fits/{fit}"
            endpoints[f"R{rotation}_{fit}"] = _durable_endpoint(directory, fit, rotation)
            label = "a5" if fit == "R_A5" else "c2f"
            producer_prefix = f"roots/r{rotation}_{label}"
            _add_tree(
                records,
                f"{producer_prefix}/rotations/{rotation}/fits/{fit}",
                directory,
            )
            for name in (
                "MIGRATION_BINDING.json",
                "SCIENTIFIC_CONFIG.json",
                "SCIENTIFIC_CLOUD_ADMISSION.json",
            ):
                _add_record(
                    records, f"{producer_prefix}/{name}", producer / name, producer
                )
            proof = producer / f"technical_resume_probe/{fit}/RESULT.json"
            _add_record(
                records,
                f"{producer_prefix}/technical_resume_probe/{fit}/RESULT.json",
                proof,
                producer,
            )
            authorization = (
                authorization_root / f"r{rotation}_c2f"
                if fit == "R_C2F"
                else authorization_root / f"r{rotation}"
            )
            if fit == "R_A5" and rotation == 4 and not authorization.is_dir():
                authorization = authorization_root
            assignment = (
                f"SCIENTIFIC_R{rotation}_C2F.json"
                if fit == "R_C2F"
                else f"SCIENTIFIC_R{rotation}.json"
            )
            for name in (
                assignment,
                "USER_BUDGET_24USD.json",
                "LOCAL_QUEUE_EXCLUSION.json",
                "WINDOWS_CHECKPOINT_COMPATIBILITY.json",
            ):
                source = authorization / name
                _add_record(
                    records,
                    f"{producer_prefix}/cloud_authorization/{name}",
                    source,
                    authorization,
                )
            ram_admission = producer / "RAM_POLICY_ADMISSION.json"
            if ram_admission.is_file():
                _add_record(
                    records,
                    f"{producer_prefix}/RAM_POLICY_ADMISSION.json",
                    ram_admission,
                    producer,
                )
            required_sources = {
                "runpod_technical_overlay_v3.py",
                "runpod_resources_v3.py",
                (
                    "runpod_scientific_overlay_c2f_v1.py"
                    if fit == "R_C2F"
                    else f"runpod_scientific_overlay_v{4 if rotation == 4 else 5}.py"
                ),
            }
            if ram_admission.is_file():
                required_sources.add("runpod_ram_policy_overlay_v1.py")
            quota_admission = producer / "QUOTA_RACE_POLICY_ADMISSION.json"
            if quota_admission.is_file():
                quota_source = operational_by_name.get("runpod_quota_race_overlay_v1.py")
                if quota_source is None or read_json(quota_admission).get(
                    "policy_source_sha256"
                ) != sha256(quota_source):
                    raise ValueError("Quota retry policy differs from its operational admission")
                _add_record(
                    records, f"{producer_prefix}/{quota_admission.name}", quota_admission, producer
                )
                required_sources.add("runpod_quota_race_overlay_v1.py")
            missing_sources = required_sources - set(operational_by_name)
            if missing_sources:
                raise ValueError(
                    f"R{rotation}/{fit} operational source closure is missing: "
                    f"{sorted(missing_sources)}"
                )
            for name in sorted(required_sources):
                source = operational_by_name[name]
                _add_record(
                    records,
                    f"{producer_prefix}/cloud_authorization/{name}",
                    source,
                    source.parent,
                )
            _nested_manifest(
                records,
                prefix=producer_prefix,
                rotation=rotation,
                fit=fit,
                endpoint=endpoints[f"R{rotation}_{fit}"],
            )
        for fit in ("PAIR_R", "R_CTX"):
            endpoints[f"R{rotation}_{fit}"] = _durable_endpoint(
                downstream / f"rotations/{rotation}/fits/{fit}", fit, rotation
            )
        if result.get("artifacts") != read_json(downstream / "RGB_DOWNSTREAM_RESULT.json").get(
            "artifacts"
        ):
            raise ValueError("RGB result changed while its artifacts were inspected")
    coverage_receipt = coverage_receipt.resolve(strict=True)
    coverage = read_json(coverage_receipt)
    if (
        coverage.get("schema") != "ejepa_rgb_downstream_bundle_coverage_v1"
        or coverage.get("status") != "PASS"
        or not set(map(str, selected)).issubset(set(coverage.get("rotations", {})))
    ):
        raise ValueError("RGB downstream coverage is incomplete")
    _add_record(
        records,
        "operations/RGB_DOWNSTREAM_COVERAGE.json",
        coverage_receipt,
        coverage_receipt.parent,
    )
    if not operational_sources:
        raise ValueError("Operational source closure must be explicit")
    source_names: set[str] = set()
    for source in operational_sources:
        source = source.resolve(strict=True)
        if source.name in source_names:
            raise ValueError(f"Duplicate operational source basename: {source.name}")
        source_names.add(source.name)
        _add_record(records, f"operations/source/{source.name}", source, source.parent)
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "status": "READY",
        "created_utc": datetime.now(UTC).isoformat(),
        "rotations": list(selected),
        "campaign_complete": selected == ROTATIONS,
        "individual_rotation_return": selected == INDIVIDUAL_RETURN_ROTATIONS,
        "complete_rgb_oof_only": True,
        "progress_presented_as_endpoint": False,
        "event_branch_included": False,
        "endpoints": endpoints,
        "input_bindings": {
            "RGB_PILOT.tar": {
                "sha256": PILOT_SHA256,
                "included": False,
                "reason": "already present and verified locally; avoid duplicate >10GB input",
            }
        },
        "source_paths_preserved_verbatim": True,
        "unknown_symlinks_followed": False,
        "active_rgb_context_chain": (
            ["V5", "V6", "V7", "V8"]
            if compositional_v8_rotations
            else ["V5", "V6", "V7"]
        ),
        "compositional_v8_rotations": sorted(compositional_v8_rotations),
        "historical_v4_preserved_not_active": sorted(historical_v4_rotations),
        "rgb_media_scope_evidence": media_scope_evidence,
        "sparse_rgb_tar_payloads_included": False,
        "automatic_cleanup_implemented": False,
        "files": records,
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest, records


def publish_rgb_return(
    manifest: dict[str, Any], records: Mapping[str, Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    """Write a regular-file-only tar, then prove its sources did not change."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing RGB archive: {output}")
    receipt = output.with_name(output.name + ".json")
    if receipt.exists():
        raise FileExistsError(f"Refusing to replace an existing RGB archive receipt: {receipt}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.pending")
    try:
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        def entry_bytes(name: str, size: int) -> int:
            info = tarfile.TarInfo(name)
            info.size = size
            headers = info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape")
            blocks = (size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
            return len(headers) + blocks * tarfile.BLOCKSIZE

        predicted = entry_bytes("MANIFEST.json", len(manifest_bytes)) + tarfile.RECORDSIZE
        predicted += sum(
            entry_bytes(relative, int(identity["bytes"]))
            for relative, identity in records.items()
        )
        predicted = (
            (predicted + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
        ) * tarfile.RECORDSIZE
        # On Linux shutil.disk_usage delegates to statvfs for the exact output
        # mount (for example /opt overlay), not the Network Volume quota report.
        available = shutil.disk_usage(output.parent).free
        required = predicted + MIN_ARCHIVE_RESERVE_BYTES
        if available < required:
            raise ValueError(
                "Insufficient real output-filesystem capacity for RGB tar plus 2 GiB reserve: "
                f"available={available}, predicted_tar={predicted}, required={required}"
            )
        with tarfile.open(temporary, "w") as archive:
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for relative, identity in sorted(records.items()):
                inline = identity.get("inline_utf8")
                if isinstance(inline, str):
                    payload = inline.encode("utf-8")
                    if (
                        len(payload) != identity["bytes"]
                        or hashlib.sha256(payload).hexdigest() != identity["sha256"]
                    ):
                        raise ValueError(f"Generated archive metadata changed: {relative}")
                    info = tarfile.TarInfo(relative)
                    info.size = len(payload)
                    info.mode = 0o444
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))
                    continue
                source = Path(str(identity["source_path"]))
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"Archive source is no longer a regular file: {source}")
                stat = source.stat()
                if stat.st_size != identity["bytes"] or sha256(source) != identity["sha256"]:
                    raise ValueError(f"Archive source changed before packing: {source}")
                info = tarfile.TarInfo(relative)
                info.size = stat.st_size
                info.mode = 0o444
                info.mtime = 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        for identity in records.values():
            if identity.get("generated_archive_metadata") is True:
                continue
            source = Path(str(identity["source_path"]))
            if source.stat().st_size != identity["bytes"] or sha256(source) != identity["sha256"]:
                raise ValueError(f"Archive source changed during packing: {source}")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "schema": "ejepa_cloud_rgb_complete_export_v1",
        "status": "READY",
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
        "predicted_archive_bytes": predicted,
        "output_filesystem_available_before_bytes": available,
        "output_filesystem_reserve_bytes": MIN_ARCHIVE_RESERVE_BYTES,
        "manifest_content_sha256": manifest["content_sha256"],
        "files": len(records),
        "training_stopped": False,
        "remote_cleanup_performed": False,
    }
    result["receipt"] = str(receipt)
    pending_receipt = receipt.with_name(f".{receipt.name}.{os.getpid()}.pending")
    pending_receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pending_receipt.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(pending_receipt, receipt)
    return result


def export_if_ready(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    output: Path,
    rotations: Sequence[int] = ROTATIONS,
    writer_probe: Callable[[Sequence[Path]], list[dict[str, object]]] = _writer_fds_linux,
) -> dict[str, Any]:
    """Return NOT_READY without publishing unless the complete closure is quiescent."""
    selected = tuple(rotations)
    roots = [
        mapping[rotation]
        for rotation in selected
        for mapping in (downstream_runs, a5_runs, c2f_runs)
    ]
    try:
        resolved_output = output.resolve()
        protected = [*roots, authorization_root]
        if any(resolved_output.is_relative_to(root.resolve(strict=True)) for root in protected):
            raise ValueError("RGB archive output must remain outside every live/input root")
        writers = writer_probe(roots)
        if writers:
            return {"status": "NOT_READY", "reason": "ACTIVE_WRITER", "writers": writers}
        manifest, records = collect_rgb_return(
            downstream_runs,
            a5_runs,
            c2f_runs,
            coverage_receipt=coverage_receipt,
            authorization_root=authorization_root,
            operational_sources=operational_sources,
            rotations=selected,
        )
        writers = writer_probe(roots)
        if writers:
            return {"status": "NOT_READY", "reason": "ACTIVE_WRITER", "writers": writers}
        return publish_rgb_return(manifest, records, output)
    except (FileNotFoundError, ValueError) as error:
        return {"status": "NOT_READY", "reason": f"{type(error).__name__}: {error}"}


def _verified_manifest(
    archive: Path,
    expected_sha256: str,
    *,
    expected_rotations: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[tarfile.TarInfo]]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or sha256(archive) != expected_sha256:
        raise ValueError("RGB return archive SHA-256 differs")
    with tarfile.open(archive, "r") as source:
        members = source.getmembers()
        names: set[str] = set()
        for member in members:
            relative = _safe_relative(member.name)
            if not member.isfile() or relative.as_posix() in names:
                raise ValueError("RGB archive contains a special or duplicate member")
            names.add(relative.as_posix())
        payload = source.extractfile("MANIFEST.json")
        if payload is None:
            raise ValueError("RGB archive manifest is missing")
        manifest = json.load(payload)
    rotations = manifest.get("rotations")
    selected = tuple(rotations) if isinstance(rotations, list) else ()
    if selected not in {ROTATIONS, INDIVIDUAL_RETURN_ROTATIONS}:
        raise ValueError("RGB return manifest has an unsupported rotation closure")
    if expected_rotations is not None and selected != tuple(expected_rotations):
        raise ValueError("RGB return rotations differ from the requested closure")
    if (
        manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("status") != "READY"
        or manifest.get("campaign_complete") is not (selected == ROTATIONS)
        or manifest.get("individual_rotation_return")
        is not (selected == INDIVIDUAL_RETURN_ROTATIONS)
        or manifest.get("progress_presented_as_endpoint") is not False
        or manifest.get("input_bindings", {}).get("RGB_PILOT.tar", {}).get("sha256")
        != PILOT_SHA256
        or manifest.get("active_rgb_context_chain") != ["V5", "V6", "V7", "V8"]
        or manifest.get("compositional_v8_rotations") != [4]
        or manifest.get("sparse_rgb_tar_payloads_included") is not False
    ):
        raise ValueError("RGB return manifest is not a complete READY archive")
    media = manifest.get("rgb_media_scope_evidence")
    expected_scopes = {
        f"R{rotation}_{role}"
        for rotation in selected
        for role in ("refiner", "generation")
    }
    if not isinstance(media, dict) or set(media) != expected_scopes:
        raise ValueError("RGB return lacks the selected media scope evidence records")
    for scope, evidence in media.items():
        compositional = evidence.get("admission_schema") == (
            "ejepa_rgb_compositional_scope_admission_v8"
        )
        if (
            not isinstance(evidence, dict)
            or not isinstance(evidence.get("physical_root"), str)
            or not PurePosixPath(evidence["physical_root"]).is_absolute()
            or evidence.get("tar_payloads_included") is not False
            or not isinstance(evidence.get("tar_receipts"), list)
        ):
            raise ValueError(f"Malformed RGB media evidence: {scope}")
        if compositional:
            if (
                not isinstance(evidence.get("archived_genealogy"), dict)
                or evidence.get("composition_mode") not in {"DIRECT", "REUSE"}
                or evidence.get("physical_rotation") not in {3, 4}
                or evidence.get("physical_role") not in {"refiner", "generation"}
            ):
                raise ValueError(f"Malformed compositional RGB evidence: {scope}")
        elif (
            not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("plan_sha256")))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(evidence.get("manifest_sha256"))
            )
        ):
            raise ValueError(f"Malformed direct RGB media evidence: {scope}")
    files = manifest.get("files")
    if not isinstance(files, dict) or names != set(files) | {"MANIFEST.json"}:
        raise ValueError("RGB archive coverage differs from its per-file manifest")
    expected_endpoints = {
        f"R{rotation}_{fit}"
        for rotation in selected
        for fit in ("R_A5", "R_C2F", "PAIR_R", "R_CTX")
    }
    if set(manifest.get("endpoints", {})) != expected_endpoints:
        raise ValueError("RGB return endpoint set differs from selected rotations")
    for relative in files:
        match = re.match(r"roots/r([234])_(?:a5|c2f|downstream)(?:/|$)", relative)
        if match and int(match.group(1)) not in selected:
            raise ValueError("RGB return contains a root outside selected rotations")
    members_by_name = {member.name: member for member in members}
    for relative, identity in files.items():
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("bytes"), int)
            or identity["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", "")))
            or not isinstance(identity.get("source_path"), str)
        ):
            raise ValueError(f"Malformed RGB file identity: {relative}")
        raw_source = PurePosixPath(identity["source_path"])
        if not raw_source.is_absolute() or ".." in raw_source.parts:
            raise ValueError(f"RGB source path is not an original absolute Linux path: {relative}")
        if members_by_name[relative].size != identity["bytes"]:
            raise ValueError(f"RGB tar member size differs from manifest: {relative}")
    content_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("content_sha256") != content_sha256:
        raise ValueError("RGB archive manifest content binding differs")
    return manifest, members


def _download(source: str, destination: Path) -> None:
    parsed = urlparse(source)
    with destination.open("xb") as output:
        local = Path(source)
        if parsed.scheme not in {"https", "http"} and (
            not parsed.scheme or local.is_absolute() or local.drive
        ):
            with local.resolve(strict=True).open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
        elif parsed.scheme in {"https", "http"}:
            with urllib.request.urlopen(source) as response:  # noqa: S310 - explicit CLI source
                shutil.copyfileobj(response, output)
        else:
            raise ValueError(
                "Only local files and HTTP(S) are supported; SSH is intentionally absent"
            )
        output.flush()
        os.fsync(output.fileno())


def admit_local_rgb_return(
    source: str,
    *,
    expected_sha256: str,
    destination_root: Path,
    repository: Path,
    reuse_verification: Path | None = None,
    expected_rotations: Sequence[int] = ROTATIONS,
) -> dict[str, Any]:
    """Download/copy, safely extract, verify, and report cleanup ineligibility/eligibility."""
    repository = repository.resolve(strict=True)
    selected = tuple(expected_rotations)
    if selected == ROTATIONS:
        allowed = repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb"
    elif selected == INDIVIDUAL_RETURN_ROTATIONS:
        allowed = repository / "artifacts/runpod_returns/irtz8y3sigd621/individual_r4"
    else:
        raise ValueError("Local RGB admission supports only [2,3,4] or individual [4]")
    destination_root = destination_root.resolve()
    if destination_root != allowed.resolve():
        raise ValueError("RGB returns must use the dedicated final_rgb/individual_r4 directory")
    active = repository / "artifacts/final_refit_20261019_scientific_oof"
    if destination_root == active.resolve() or destination_root.is_relative_to(active.resolve()):
        raise ValueError("RGB return cannot replace or enter active fits")
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary_archive = destination_root / f".{expected_sha256}.tar.pending"
    temporary_archive.unlink(missing_ok=True)
    try:
        _download(source, temporary_archive)
        manifest, members = _verified_manifest(
            temporary_archive,
            expected_sha256,
            expected_rotations=selected,
        )
        label = "rgb_complete" if selected == ROTATIONS else "rgb_individual_r4"
        identity = f"{label}_{expected_sha256[:16]}"
        target = destination_root / identity
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ValueError("Existing RGB return target is unsafe")
            verify_extracted_rgb_return(target, expected_sha256)
        else:
            pending = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=destination_root))
            try:
                with tarfile.open(temporary_archive, "r") as archive:
                    for member in members:
                        relative = _safe_relative(member.name)
                        path = pending.joinpath(*relative.parts)
                        if not path.resolve().is_relative_to(pending.resolve()):
                            raise ValueError("RGB archive member escaped extraction root")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        payload = archive.extractfile(member)
                        if payload is None:
                            raise ValueError(f"Missing archive payload: {member.name}")
                        with path.open("xb") as output:
                            shutil.copyfileobj(payload, output)
                            output.flush()
                            os.fsync(output.fileno())
                _verify_extracted_files(pending, manifest)
                os.replace(pending, target)
            finally:
                if pending.exists():
                    shutil.rmtree(pending)
        verification = {
            "schema": "ejepa_local_rgb_complete_verification_v1",
            "status": "PASS",
            "archive_sha256": expected_sha256,
            "archive_bytes": temporary_archive.stat().st_size,
            "snapshot_root": str(target),
            "manifest_content_sha256": manifest["content_sha256"],
            "files": len(manifest["files"]),
            "rotations": list(selected),
            "active_fits_replaced": False,
            "raw_linux_paths_rewritten": False,
        }
        (target / "LOCAL_VERIFICATION.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if reuse_verification is None:
            cleanup = {
                "status": "NOT_EVALUATED",
                "reason": "SCIENTIFIC_REUSABILITY_VERIFIER_NOT_PROVIDED",
                "report_emitted": False,
                "automatic_cleanup_available": False,
                "remote_deletion_performed": False,
            }
        else:
            reuse = read_json(reuse_verification.resolve(strict=True))
            reuse_sha = sha256(reuse_verification)
            rotations = reuse.get("rotations")
            expected_rotation_keys = set(map(str, selected))
            endpoint_evidence = (
                isinstance(rotations, dict) and set(rotations) == expected_rotation_keys
            )
            if endpoint_evidence and isinstance(rotations, dict):
                for rotation in sorted(expected_rotation_keys):
                    rotation_evidence = rotations[rotation]
                    endpoints = (
                        rotation_evidence.get("endpoints", {})
                        if isinstance(rotation_evidence, dict)
                        else {}
                    )
                    endpoint_evidence = isinstance(endpoints, dict) and set(endpoints) == {
                        "PAIR_R",
                        "R_CTX",
                    } and all(
                        endpoint.get("original_verify_scientific_admission") == "PASS"
                        and re.fullmatch(
                            r"[0-9a-f]{64}", str(endpoint.get("admission_sha256", ""))
                        )
                        and re.fullmatch(
                            r"[0-9a-f]{64}", str(endpoint.get("checkpoint_sha256", ""))
                        )
                        for endpoint in endpoints.values()
                    )
                    if not endpoint_evidence:
                        break
            eligible = (
                reuse.get("schema") == "ejepa_local_rgb_scientific_reusability_v1"
                and reuse.get("status") == "PASS"
                and reuse.get("scientific_reusability_verified") is True
                and reuse.get("archive_sha256") == expected_sha256
                and reuse.get("selected_root_path_maps_verified") is True
                and (
                    selected != ROTATIONS
                    or reuse.get("three_root_path_maps_verified") is True
                )
                and reuse.get("active_fits_modified") is False
                and endpoint_evidence
            )
            blockers = [] if eligible else ["SCIENTIFIC_REUSABILITY_VERIFIER_DID_NOT_PASS"]
            cleanup = {
                "schema": "ejepa_rgb_remote_cleanup_eligibility_v1",
                "status": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
                "archive_sha256": expected_sha256,
                "local_complete_verification": "PASS",
                "scientific_reusability_verification_sha256": reuse_sha,
                "blockers": blockers,
                "report_emitted": True,
                "automatic_cleanup_available": False,
                "remote_deletion_performed": False,
                "local_deletion_performed": False,
                "requires_explicit_separate_cleanup_action": True,
            }
            (target / "CLEANUP_ELIGIBILITY.json").write_text(
                json.dumps(cleanup, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        eviction: dict[str, Any] | None = None
        if selected == INDIVIDUAL_RETURN_ROTATIONS and reuse_verification is not None:
            eviction = {
                "schema": "ejepa_rgb_raw_rotation_eviction_eligibility_v1",
                "status": "PASS" if cleanup["status"] == "ELIGIBLE" else "NOT_ELIGIBLE",
                "rotation": 4,
                "scope": "RAW_R4_REPRODUCIBLE",
                "archive_sha256": expected_sha256,
                "archive_manifest_sha256": sha256(target / "MANIFEST.json"),
                "local_verification_sha256": sha256(target / "LOCAL_VERIFICATION.json"),
                "scientific_reusability_verification_sha256": sha256(
                    reuse_verification.resolve(strict=True)
                ),
                "scientific_genealogy_verified": cleanup["status"] == "ELIGIBLE",
                "raw_rgb_payloads_archived": False,
                "raw_rgb_reproducible_from_locally_verified_originals": (
                    cleanup["status"] == "ELIGIBLE"
                ),
                "external_uploader_may_use_as_eviction_authority": (
                    cleanup["status"] == "ELIGIBLE"
                ),
                "remote_deletion_performed": False,
                "local_deletion_performed": False,
                "automatic_deletion_implemented": False,
                "requires_separate_explicit_external_action": True,
            }
            eviction_path = destination_root / "RAW_R4_EVICTION_ELIGIBILITY.json"
            eviction_path.write_text(
                json.dumps(eviction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return {**verification, "cleanup": cleanup, "eviction": eviction}
    finally:
        temporary_archive.unlink(missing_ok=True)


def _verify_extracted_files(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = manifest["files"]
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != set(expected) | {"MANIFEST.json"}:
        raise ValueError("Extracted RGB file coverage differs")
    for relative, identity in expected.items():
        path = root.joinpath(*_safe_relative(relative).parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve()):
            raise ValueError(f"Extracted RGB member escaped: {relative}")
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Extracted RGB member differs: {relative}")


def verify_extracted_rgb_return(root: Path, archive_sha256: str) -> dict[str, Any]:
    """Re-verify an existing isolated return without trusting prior reports."""
    manifest = read_json(root / "MANIFEST.json")
    if manifest.get("schema") != ARCHIVE_SCHEMA or manifest.get("status") != "READY":
        raise ValueError("Existing RGB return manifest is invalid")
    ignored = {"LOCAL_VERIFICATION.json", "CLEANUP_ELIGIBILITY.json"}
    expected = set(manifest.get("files", {})) | {"MANIFEST.json"}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }
    if observed != expected:
        raise ValueError("Existing RGB return has missing or extra archive files")
    for relative, identity in manifest["files"].items():
        path = root.joinpath(*_safe_relative(relative).parts)
        if (
            path.is_symlink()
            or path.stat().st_size != identity["bytes"]
            or sha256(path) != identity["sha256"]
        ):
            raise ValueError(f"Existing RGB return member differs: {relative}")
    return {"status": "PASS", "archive_sha256": archive_sha256}


__all__ = [
    "ARCHIVE_SCHEMA",
    "PILOT_SHA256",
    "admit_local_rgb_return",
    "collect_rgb_return",
    "export_if_ready",
    "publish_rgb_return",
    "sha256",
    "verify_extracted_rgb_return",
]
