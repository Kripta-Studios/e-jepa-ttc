"""Reject corrupted or unsafe migration inputs before any scientific work."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.runpod_final_phase_bridge import digest, extract_verified


def archive(path: Path, name: str = "fold/fragment.npz") -> str:
    payload = b"sealed fixture"
    manifest = {
        "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    }
    with tarfile.open(path, "w") as tar:
        for key, data in [("MANIFEST.json", json.dumps(manifest).encode()), (name, payload)]:
            member = tarfile.TarInfo(key)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return digest(path)


def test_verified_extraction_resumes_without_overwriting(tmp_path):
    source = tmp_path / "input.tar"
    expected = archive(source)
    out = tmp_path / "out"
    extract_verified(source, out, expected)
    stat = (out / "fold/fragment.npz").stat()
    extract_verified(source, out, expected)
    assert (out / "fold/fragment.npz").stat().st_mtime_ns == stat.st_mtime_ns


def test_rejects_modified_existing_input(tmp_path):
    source = tmp_path / "input.tar"
    expected = archive(source)
    out = tmp_path / "out"
    extract_verified(source, out, expected)
    (out / "fold/fragment.npz").write_bytes(b"other fixture")
    with pytest.raises(ValueError, match="changed"):
        extract_verified(source, out, expected)


def test_rejects_archive_traversal(tmp_path):
    source = tmp_path / "input.tar"
    expected = archive(source, "../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_verified(source, tmp_path / "out", expected)
    assert not (tmp_path / "outside").exists()


def test_rejects_wrong_archive_identity(tmp_path):
    source = tmp_path / "input.tar"
    archive(source)
    with pytest.raises(ValueError, match="checksum"):
        extract_verified(source, tmp_path / "out", "0" * 64)
