"""Pin an immutable checkpoint version and matching receipts without pausing training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def publish_archive(target: Path, archive: Path) -> None:
    """Recover an interrupted archive publication from verified pinned files."""
    manifest = json.loads((target / "MANIFEST.json").read_text())
    expected = set(manifest["files"]) | {"MANIFEST.json"}
    observed = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    if observed != expected:
        raise ValueError("Published snapshot has missing or extra files")
    for relative, identity in manifest["files"].items():
        path = target / relative
        if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
            raise ValueError("Snapshot file escaped its pinned root")
        if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            raise ValueError("Published snapshot file differs from its manifest")
    temporary = archive.with_suffix(".tar.pending")
    with tarfile.open(temporary, "w", dereference=True) as stream:
        for relative in sorted(expected):
            stream.add(target / relative, arcname=relative)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, archive)
    if os.name != "nt":
        descriptor = os.open(archive.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def snapshot(run: Path, rotation: int, fit: str) -> dict:
    """Only publish when pointer/receipt agree; journal records observed recovery exposure."""
    if rotation not in {2, 3, 4} or fit != "R_C2F":
        raise ValueError("Only assigned RGB C2F tasks may be exported")
    directory = run / "rotations" / str(rotation) / "fits" / fit
    names = ("CHECKPOINT_POINTER.json", "CHECKPOINT_RECEIPT.json", "UPDATE_JOURNAL.json")
    try:
        raw = {name: (directory / name).read_bytes() for name in names}
    except FileNotFoundError:
        return {"status": "NOT_READY"}
    pointer, receipt, journal = (json.loads(raw[name]) for name in names)
    step = pointer["completed_updates"]
    if (
        step < 0
        or pointer["version"] != f"checkpoint_{step:06d}.pt"
        or receipt["completed_updates"] != step
        or receipt["checkpoint_sha256"] != pointer["checkpoint_sha256"]
        or receipt["identity_sha256"] != pointer["identity_sha256"]
        or journal["durable_updates"] != step
        or journal["completed_updates"] < step
    ):
        return {"status": "NOT_READY", "reason": "CHECKPOINT_TRANSACTION_IN_FLIGHT"}
    identity = f"{step:06d}_{pointer['checkpoint_sha256'][:12]}"
    root = run / "cloud_snapshots" / f"r{rotation}_{fit}"
    root.mkdir(parents=True, exist_ok=True)
    target, archive = root / identity, root / f"{identity}.tar"
    if target.joinpath("MANIFEST.json").is_file():
        if not archive.exists():
            publish_archive(target, archive)
        return {
            "status": "READY",
            "snapshot": identity,
            "updates": step,
            "archive": str(archive),
            "archive_sha256": sha(archive),
        }
    pending = root / f".{identity}.{os.getpid()}.pending"
    pending.mkdir()
    try:
        relative = Path("rotations") / str(rotation) / "fits" / fit
        copied = pending / relative
        (copied / "checkpoint_versions").mkdir(parents=True)
        version = copied / "checkpoint_versions" / pointer["version"]
        source = directory / "checkpoint_versions" / pointer["version"]
        try:
            os.link(source, version)
        except OSError:
            shutil.copyfile(source, version)
        if sha(version) != pointer["checkpoint_sha256"]:
            raise ValueError("Pinned immutable checkpoint differs from pointer")
        # Receipt/alias are never followed as the data source. Both copies come
        # from the pinned immutable version, so a later pointer cannot change them.
        shutil.copyfile(version, copied / "checkpoint_last.pt")
        for name, value in raw.items():
            (copied / name).write_bytes(value)
        if any((directory / name).read_bytes() != raw[name] for name in names[:2]):
            return {"status": "NOT_READY", "reason": "POINTER_ADVANCED_DURING_SNAPSHOT"}
        for name in (
            "MIGRATION_BINDING.json",
            "SCIENTIFIC_CONFIG.json",
            "SCIENTIFIC_CLOUD_ADMISSION.json",
        ):
            shutil.copyfile(run / name, pending / name)
        proof = pending / "technical_resume_probe/R_C2F"
        proof.mkdir(parents=True)
        shutil.copyfile(run / "technical_resume_probe/R_C2F/RESULT.json", proof / "RESULT.json")
        authorization = pending / "cloud_authorization"
        authorization.mkdir()
        authorization_source = Path("/workspace/ejepa_operations")
        authorization_source /= f"r{rotation}_c2f"
        for name in (
            f"SCIENTIFIC_R{rotation}_C2F.json",
            "USER_BUDGET_24USD.json",
            "LOCAL_QUEUE_EXCLUSION.json",
            "WINDOWS_CHECKPOINT_COMPATIBILITY.json",
        ):
            shutil.copyfile(authorization_source / name, authorization / name)
        for name in (
            "runpod_scientific_overlay_c2f_v1.py",
            "runpod_technical_overlay_v3.py",
            "runpod_resources_v3.py",
        ):
            shutil.copyfile(Path("/workspace/ejepa_operations") / name, authorization / name)
        ram_admission = run / "RAM_POLICY_ADMISSION.json"
        if ram_admission.is_file():
            shutil.copyfile(ram_admission, pending / ram_admission.name)
            ram_source = Path("/workspace/ejepa_operations/runpod_ram_policy_overlay_v1.py")
            ram_record = json.loads(ram_admission.read_text())
            if sha(ram_source) != ram_record["policy_source_sha256"]:
                raise ValueError("RAM policy source differs from its operational admission")
            shutil.copyfile(ram_source, authorization / ram_source.name)
        quota_admission = run / "QUOTA_RACE_POLICY_ADMISSION.json"
        if quota_admission.is_file():
            quota_source = Path("/workspace/ejepa_operations/runpod_quota_race_overlay_v1.py")
            quota_record = json.loads(quota_admission.read_text())
            if sha(quota_source) != quota_record["policy_source_sha256"]:
                raise ValueError("Quota retry source differs from its operational admission")
            shutil.copyfile(quota_admission, pending / quota_admission.name)
            shutil.copyfile(quota_source, authorization / quota_source.name)
        if receipt["status"] == "COMPLETE":
            for name in ("ENDPOINT_ADMISSION.json", "STRUCTURAL_LINEAGE.json"):
                if not (directory / name).exists():
                    return {"status": "NOT_READY", "reason": "ENDPOINT_NOT_YET_SEALED"}
                shutil.copyfile(directory / name, copied / name)
        files = {}
        for path in pending.rglob("*"):
            if path.is_file():
                with path.open("rb+") as stream:
                    os.fsync(stream.fileno())
                files[path.relative_to(pending).as_posix()] = {
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                }
        manifest = {
            "schema": "ejepa_cloud_checkpoint_snapshot_v1",
            "snapshot_exporter_sha256": sha(Path(__file__)),
            "snapshot": identity,
            "rotation": rotation,
            "fit": fit,
            "completed_updates": step,
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "checkpoint_status": receipt["status"],
            "files": files,
            "observed_journal_updates": journal["completed_updates"],
            "observed_journal_pending_update_upper": journal["pending_update_upper"],
            "live_journal_captured_not_reconstructed": True,
            "training_paused_for_snapshot": False,
        }
        (pending / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(pending, target)
        publish_archive(target, archive)
        published = sorted(
            [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")],
            key=lambda path: path.name,
            reverse=True,
        )
        for stale in published[2:]:
            old_manifest = json.loads((stale / "MANIFEST.json").read_text())
            if old_manifest["checkpoint_status"] == "COMPLETE":
                continue
            if not stale.resolve().is_relative_to(root.resolve()) or stale.is_symlink():
                raise ValueError("Snapshot retention target escaped its dedicated root")
            shutil.rmtree(stale)
            stale.with_suffix(".tar").unlink(missing_ok=True)
        return {
            "status": "READY",
            "snapshot": identity,
            "updates": step,
            "archive": str(archive),
            "archive_sha256": sha(archive),
        }
    finally:
        if pending.exists():
            shutil.rmtree(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run.resolve(strict=True), args.rotation, "R_C2F")))
