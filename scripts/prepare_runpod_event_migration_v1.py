"""Prepare a zero-deployment, event-only Runpod migration handoff.

The command reads the already-built FULL_V2 inventory.  It never starts a Pod,
uploads data, pauses a trainer, or authorizes optimizer updates.  Bulk event
inputs are emitted as independently resumable manifests; only the small control
plane is packed into a tar archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

GIB = 1024**3
EVENT_FITS = ("E_A5", "E_C2F")
ROTATIONS = (2, 3, 4)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    with pending.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _inside(path: str, root: str) -> bool:
    candidate, parent = PureWindowsPath(path), PureWindowsPath(root)
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _manifest(schema: str, profile: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        {
            "source": str(item["source"]),
            "relative": str(item["relative"]),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
            "hash_status": str(item["hash_status"]),
            "hash_provenance": str(item.get("hash_provenance", "")),
        }
        for item in sorted(files, key=lambda row: str(row["relative"]))
    ]
    if any(len(item["sha256"]) != 64 for item in selected):
        raise ValueError(f"{profile} contains an unbound file")
    identity_rows = [
        {key: item[key] for key in ("relative", "bytes", "sha256")} for item in selected
    ]
    return {
        "schema": schema,
        "profile": profile,
        "created_utc": datetime.now(UTC).isoformat(),
        "files": selected,
        "file_count": len(selected),
        "total_bytes": sum(item["bytes"] for item in selected),
        "total_gib": round(sum(item["bytes"] for item in selected) / GIB, 3),
        "content_identity_sha256": _canonical_sha(identity_rows),
        "all_files_content_bound": True,
        "transfer_complete": False,
    }


def _checkpoint_observation(run: Path) -> dict[str, Any]:
    directory = run / "rotations/2/fits/E_A5"
    pointer = _read(directory / "CHECKPOINT_POINTER.json")
    receipt = _read(directory / "CHECKPOINT_RECEIPT.json")
    before = _read(directory / "UPDATE_JOURNAL.json")
    version = directory / "checkpoint_versions" / str(pointer["version"])
    actual = _digest(version)
    after = _read(directory / "UPDATE_JOURNAL.json")
    identity = str(pointer["identity_sha256"])
    coherent = bool(
        actual == pointer.get("checkpoint_sha256") == receipt.get("checkpoint_sha256")
        and pointer.get("completed_updates") == receipt.get("completed_updates")
        and pointer.get("completed_updates") == after.get("durable_updates")
        and identity == receipt.get("identity_sha256") == after.get("identity_sha256")
        and int(after.get("pending_update_upper", -1)) in {0, 1}
    )
    if not coherent:
        raise ValueError("E_A5/R2 durable checkpoint is not internally coherent")
    pause_path = Path(__file__).resolve().parents[1] / (
        "artifacts/runpod_migration_20261026/LOCAL_EVENT_HANDOFF_PAUSE.json"
    )
    pause = _read(pause_path) if pause_path.is_file() else None
    authoritative = bool(
        pause
        and pause.get("status") == "PAUSED_DURABLY_EVENT_POD_HANDOFF_PENDING"
        and pause.get("completed_updates") == pointer["completed_updates"]
        and pause.get("checkpoint_sha256") == actual
        and pause.get("updates_lost") == 0
    )
    return {
        "rotation": 2,
        "fit": "E_A5",
        "observed_utc": datetime.now(UTC).isoformat(),
        "durable_updates": int(pointer["completed_updates"]),
        "journal_before": int(before["completed_updates"]),
        "journal_after": int(after["completed_updates"]),
        "pending_update_upper": int(after["pending_update_upper"]),
        "version": pointer["version"],
        "version_bytes": version.stat().st_size,
        "checkpoint_sha256": actual,
        "identity_sha256": identity,
        "status": receipt["status"],
        "local_training_was_paused": authoritative,
        "authoritative_handoff": authoritative,
        "pause_receipt": str(pause_path) if authoritative else None,
        "pause_receipt_sha256": _digest(pause_path) if authoritative else None,
        "note": (
            "Authoritative paused handoff; cloud resume remains blocked pending execution binding."
            if authoritative
            else (
                "Observation only. Capture a newer durable boundary after cloud preflight "
                "and local ownership exclusion."
            )
        ),
    }


def _pack_handoff(run: Path, output: Path, checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    if not checkpoint["authoritative_handoff"]:
        return None
    directory = run / "rotations/2/fits/E_A5"
    version_name = str(checkpoint["version"])
    source_files = {
        f"rotations/2/fits/E_A5/checkpoint_versions/{version_name}": directory
        / "checkpoint_versions"
        / version_name,
        "rotations/2/fits/E_A5/CHECKPOINT_POINTER.json": directory / "CHECKPOINT_POINTER.json",
        "rotations/2/fits/E_A5/CHECKPOINT_RECEIPT.json": directory / "CHECKPOINT_RECEIPT.json",
        "rotations/2/fits/E_A5/UPDATE_JOURNAL.json": directory / "UPDATE_JOURNAL.json",
        "rotations/2/fits/E_A5/PROGRESS.json": directory / "PROGRESS.json",
        "handoff_authority/LOCAL_EVENT_HANDOFF_PAUSE.json": Path(checkpoint["pause_receipt"]),
    }
    bindings = {
        relative: {"bytes": path.stat().st_size, "sha256": _digest(path)}
        for relative, path in source_files.items()
    }
    manifest = {
        "schema": "ejepa_event_checkpoint_handoff_v1",
        "rotation": 2,
        "fit": "E_A5",
        "completed_updates": checkpoint["durable_updates"],
        "endpoint_updates": 29_574,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "files": bindings,
        "local_stop_marker_transferred_to_cloud_fit": False,
        "resume_authorized": False,
        "cross_hardware_numerical_continuity_claimed": False,
    }
    manifest["payload_sha256"] = _canonical_sha(manifest)
    archive = output / "E_A5_R2_HANDOFF.tar"
    if archive.exists():
        raise FileExistsError(archive)
    pending = archive.with_suffix(".tar.partial")
    try:
        with tarfile.open(pending, "w") as stream:
            for relative, path in source_files.items():
                info = stream.gettarinfo(str(path), arcname=relative)
                with path.open("rb") as source:
                    stream.addfile(info, source)
            payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo("HANDOFF_MANIFEST.json")
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
        os.replace(pending, archive)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "archive": str(archive.resolve()),
        "bytes": archive.stat().st_size,
        "sha256": _digest(archive),
        "completed_updates": checkpoint["durable_updates"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "resume_authorized": False,
    }


def _pack_control(
    archive: Path,
    files: list[dict[str, Any]],
    generated: list[Path],
) -> dict[str, Any]:
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix(archive.suffix + ".partial")
    if pending.exists():
        raise FileExistsError(pending)
    try:
        with tarfile.open(pending, "w") as output:
            for item in files:
                source = Path(str(item["source"]))
                relative = PurePosixPath(str(item["relative"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe control member: {relative}")
                before = source.stat()
                actual = _digest(source)
                if before.st_size != int(item["bytes"]) or actual != item["sha256"]:
                    raise ValueError(f"Control source changed: {source}")
                info = output.gettarinfo(str(source), arcname=str(relative))
                with source.open("rb") as stream:
                    output.addfile(info, stream)
            for source in generated:
                payload = source.read_bytes()
                arcname = (
                    "BUNDLE_MANIFEST.json"
                    if source.name == "BUNDLE_MANIFEST.json"
                    else f"event_migration/{source.name}"
                )
                info = tarfile.TarInfo(arcname)
                info.size = len(payload)
                info.mtime = int(source.stat().st_mtime)
                output.addfile(info, io.BytesIO(payload))
        os.replace(pending, archive)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "archive": str(archive.resolve()),
        "bytes": archive.stat().st_size,
        "sha256": _digest(archive),
        "bulk_event_data_included": False,
        "raw_event_h5_included": False,
        "raw_rgb_included": False,
    }


def prepare(inventory_path: Path, run: Path, output: Path) -> dict[str, Any]:
    inventory_path, run = inventory_path.resolve(strict=True), run.resolve(strict=True)
    inventory = _read(inventory_path)
    config = _read(run / "SCIENTIFIC_CONFIG.json")
    event_root, raw_root = str(config["event_root"]), str(config["raw_root"])
    files = list(inventory["files"])
    event_files = [
        item
        for item in files
        if (
            _inside(str(item["source"]), event_root)
            or "/producer_inputs/" in str(item["relative"]).replace("\\", "/")
        )
        and "all_producers" in item["profiles"]
    ]
    raw_event_files = [
        item
        for item in files
        if _inside(str(item["source"]), raw_root)
        and (
            str(item["source"]).lower().endswith("events.h5")
            or str(item["source"]).lower().endswith("data\\train.parquet")
        )
    ]
    raw_rgb = [item for item in files if str(item["relative"]).lower().endswith(".tar")]
    main_repo_marker = "workspace/e-jepa-ttc-v18-joint-measurement/"
    control = [
        item
        for item in files
        if main_repo_marker in str(item["relative"])
        and "all_producers" in item["profiles"]
        and "/cache/" not in str(item["relative"]).lower()
        and "/rotations/" not in str(item["relative"]).lower()
        and not str(item["relative"]).lower().endswith((".pt", ".npz", ".tar", ".h5"))
    ]
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "scripts/prepare_runpod_event_migration_v1.py",
        "scripts/runpod_event_transfer_v1.py",
        "operational/runpod_migration/immutable_seal.py",
        "docs/runpod_migration_20261026/EVENT_MIGRATION_PLAN.md",
    ):
        path = (repo / name).resolve(strict=True)
        if not any(Path(str(item["source"])).resolve() == path for item in control):
            control.append(
                {
                    "source": str(path),
                    "relative": f"workspace/{repo.name}/{name}",
                    "bytes": path.stat().st_size,
                    "sha256": _digest(path),
                    "hash_status": "VERIFIED_AT_EVENT_PREPARATION",
                    "hash_provenance": "event migration implementation closure",
                    "profiles": ["event_producers"],
                }
            )
    if len(raw_event_files) != 41:
        raise ValueError(
            f"Expected train.parquet plus 40 event HDF5 files, got {len(raw_event_files)}"
        )
    if not 8_000 <= len(event_files) <= 9_000:
        raise ValueError(f"Unexpected event producer closure: {len(event_files)} files")
    output.mkdir(parents=True, exist_ok=True)
    producer_manifest = _manifest(
        "ejepa_event_producer_data_manifest_v1", "event_producers", event_files
    )
    raw_manifest = _manifest(
        "ejepa_event_raw_deferred_manifest_v1", "event_raw_deferred", raw_event_files
    )
    _write_json(output / "EVENT_PRODUCER_DATA_MANIFEST.json", producer_manifest)
    _write_json(output / "EVENT_RAW_DEFERRED_MANIFEST.json", raw_manifest)
    checkpoint = _checkpoint_observation(run)
    _write_json(output / "E_A5_R2_CHECKPOINT_OBSERVATION.json", checkpoint)
    budget = {
        "schema": "ejepa_event_cloud_budget_pending_v1",
        "status": "PENDING_DEPLOYMENT",
        "maximum_additional_usd": 24.0,
        "maximum_wall_hours": 24.0,
        "deployment_timestamp_utc": None,
        "execution_deadline_utc": None,
        "deadline_starts_only_after_actual_rental": True,
        "user_controls_pod_stop": True,
        "separate_from_first_pod_budget": True,
        "secrets_recorded": False,
    }
    _write_json(output / "EVENT_BUDGET_PENDING_DEPLOYMENT.json", budget)
    coverage = {
        "schema": "ejepa_event_migration_coverage_v1",
        "status": "PREPARED_NOT_DEPLOYED",
        "rotations": list(ROTATIONS),
        "producer_fits": list(EVENT_FITS),
        "downstream": ["PAIR_E", "E_CTX", "event OOF features", "final OOF stitch inputs"],
        "producer_data": {
            "files": producer_manifest["file_count"],
            "bytes": producer_manifest["total_bytes"],
            "content_identity_sha256": producer_manifest["content_identity_sha256"],
        },
        "raw_event_deferred": {
            "sequences": 40,
            "files_including_train_parquet": raw_manifest["file_count"],
            "bytes": raw_manifest["total_bytes"],
            "content_identity_sha256": raw_manifest["content_identity_sha256"],
        },
        "raw_rgb_excluded": {
            "files": len(raw_rgb),
            "bytes": sum(int(item["bytes"]) for item in raw_rgb),
            "reason": "Event branch does not consume raw RGB archives.",
        },
        "checkpoint_observation": checkpoint,
        "gates": [
            "Pod host must be measured before scientific execution.",
            "Producer inputs must be fully transferred, content-verified once, and staged locally.",
            "A zero-update input reference must match EVENT_INPUT_REFERENCE_R2/R3/R4.",
            "E_A5/R2 local ownership must be excluded only after cloud readiness.",
            "The final handoff checkpoint must be captured at a durable 100-update boundary.",
        ],
    }
    _write_json(output / "EVENT_COVERAGE.json", coverage)
    bundle_files = [
        {
            "source": str(item["source"]),
            "relative": str(item["relative"]),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
            "hash_status": str(item["hash_status"]),
            "hash_provenance": str(item.get("hash_provenance", "")),
            "profiles": ["event_producers"],
        }
        for item in sorted([*control, *event_files], key=lambda row: str(row["relative"]))
    ]
    bundle_manifest = {
        "schema": "ejepa_runpod_event_bundle_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_config_sha256": _digest(run / "SCIENTIFIC_CONFIG.json"),
        "source_run": str(run),
        "source_repo": inventory["source_repo"],
        "origin_git_commit": inventory["origin_git_commit"],
        "origin_git_commit_is_base_only": True,
        "exact_execution_code_authority": inventory["exact_execution_code_authority"],
        "roots": inventory["roots"],
        "bundle_root": "/workspace/ejepa_event_bundle",
        "profile": "event_producers",
        "profiles": {
            "event_producers": {
                "files": len(bundle_files),
                "input_bytes": sum(item["bytes"] for item in bundle_files),
                "input_gib": round(sum(item["bytes"] for item in bundle_files) / GIB, 3),
                "unhashed_files": 0,
            }
        },
        "packages": inventory["packages"],
        "files": bundle_files,
        "execution_ready": False,
        "raw_event_deferred": True,
        "raw_rgb_included": False,
    }
    _write_json(output / "BUNDLE_MANIFEST.json", bundle_manifest)
    generated = [
        output / "BUNDLE_MANIFEST.json",
        output / "EVENT_PRODUCER_DATA_MANIFEST.json",
        output / "EVENT_RAW_DEFERRED_MANIFEST.json",
        output / "E_A5_R2_CHECKPOINT_OBSERVATION.json",
        output / "EVENT_BUDGET_PENDING_DEPLOYMENT.json",
        output / "EVENT_COVERAGE.json",
    ]
    archive = output / "EVENT_CONTROL.tar"
    archive_receipt = _pack_control(archive, control, generated)
    _write_json(output / "EVENT_CONTROL.tar.json", archive_receipt)
    handoff_receipt = _pack_handoff(run, output, checkpoint)
    if handoff_receipt is not None:
        _write_json(output / "E_A5_R2_HANDOFF.tar.json", handoff_receipt)
    result = {
        "schema": "ejepa_event_migration_preparation_v1",
        "status": "PREPARED_NOT_DEPLOYED",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_inventory": str(inventory_path),
        "source_inventory_sha256": _digest(inventory_path),
        "scientific_config_sha256": _digest(run / "SCIENTIFIC_CONFIG.json"),
        "control_archive": archive_receipt,
        "checkpoint_handoff": handoff_receipt,
        "producer_manifest_sha256": _digest(output / "EVENT_PRODUCER_DATA_MANIFEST.json"),
        "bundle_manifest_sha256": _digest(output / "BUNDLE_MANIFEST.json"),
        "raw_deferred_manifest_sha256": _digest(output / "EVENT_RAW_DEFERRED_MANIFEST.json"),
        "budget_receipt_sha256": _digest(output / "EVENT_BUDGET_PENDING_DEPLOYMENT.json"),
        "deployment_timestamp_utc": None,
        "pod_created": False,
        "data_transferred": False,
        "local_training_paused": checkpoint["local_training_was_paused"],
        "scientific_optimizer_updates_authorized": 0,
    }
    _write_json(output / "PREPARATION_RECEIPT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inventory, args.run, args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
