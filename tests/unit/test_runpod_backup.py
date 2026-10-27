"""Backup archives are verified and retained without touching live training."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.runpod_backup_v1 import admit_archive, digest, retain


def archive_fixture(tmp_path, step=100, status="IN_PROGRESS"):
    raw = b"checkpoint state"
    checkpoint_hash = hashlib.sha256(raw).hexdigest()
    name = f"{step:06d}_{checkpoint_hash[:12]}"
    manifest = {
        "snapshot": name,
        "completed_updates": step,
        "checkpoint_status": status,
        "checkpoint_sha256": checkpoint_hash,
        "files": {"checkpoint.pt": {"sha256": checkpoint_hash, "bytes": len(raw)}},
    }
    path = tmp_path / f"{step}.tar"
    with tarfile.open(path, "w") as stream:
        for filename, payload in [
            ("checkpoint.pt", raw),
            ("MANIFEST.json", json.dumps(manifest).encode()),
        ]:
            entry = tarfile.TarInfo(filename)
            entry.size = len(payload)
            stream.addfile(entry, io.BytesIO(payload))
    return path, manifest


def test_verified_snapshot_publishes_and_rejects_wrong_tar_hash(tmp_path):
    archive, manifest = archive_fixture(tmp_path)
    root = tmp_path / "returns"
    root.mkdir()
    assert admit_archive(archive, root, digest(archive)) == manifest
    assert (root / manifest["snapshot"] / "checkpoint.pt").read_bytes() == b"checkpoint state"
    with pytest.raises(ValueError):
        admit_archive(archive, root, "0" * 64)


def test_unsafe_snapshot_member_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as stream:
        member = tarfile.TarInfo("../outside")
        stream.addfile(member, io.BytesIO())
    with pytest.raises(ValueError):
        admit_archive(path, tmp_path, digest(path))


def test_retention_keeps_two_latest_plus_endpoint(tmp_path):
    root = tmp_path / "returns"
    root.mkdir()
    records = []
    for step, state in [
        (100, "COMPLETE"),
        (200, "IN_PROGRESS"),
        (300, "IN_PROGRESS"),
        (400, "IN_PROGRESS"),
    ]:
        archive, manifest = archive_fixture(tmp_path, step, state)
        records.append(manifest)
        admit_archive(archive, root, digest(archive))
    retain(root)
    assert (root / records[0]["snapshot"]).exists()
    assert not (root / records[1]["snapshot"]).exists()
    assert all((root / record["snapshot"]).exists() for record in records[2:])
