"""Integrity checks for preservation archives independent of training execution."""

import builtins
import hashlib
import io
import json
import tarfile

import pytest

from scripts.runpod_final_preservation import _load_transport_call, verify_archive


def make_archive(path, *, member=b"checkpoint", member_sha=None):
    name = "workspace/fits/checkpoint.pt"
    manifest = {
        "files": {
            name: {"bytes": len(member), "sha256": member_sha or hashlib.sha256(member).hexdigest()}
        }
    }
    with tarfile.open(path, "w") as tar:
        for filename, payload in (
            (name, member),
            ("PRESERVATION_MANIFEST.json", json.dumps(manifest).encode()),
        ):
            entry = tarfile.TarInfo(filename)
            entry.size = len(payload)
            tar.addfile(entry, io.BytesIO(payload))
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_member_hashes_verified_after_archive_hash(tmp_path):
    path = tmp_path / "backup.tar"
    expected = make_archive(path)
    assert len(verify_archive(path, expected)["files"]) == 1


def test_correct_archive_hash_cannot_hide_bad_member_manifest(tmp_path):
    path = tmp_path / "backup.tar"
    expected = make_archive(path, member_sha="0" * 64)
    with pytest.raises(ValueError, match="Member SHA mismatch"):
        verify_archive(path, expected)


def test_archive_corruption_rejected(tmp_path):
    path = tmp_path / "backup.tar"
    expected = make_archive(path)
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Archive SHA mismatch"):
        verify_archive(path, expected)


def test_remote_transport_has_actionable_error_when_rpc_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def import_without_rpc(name, *args, **kwargs):
        if name == "rpc":
            raise ModuleNotFoundError("No module named 'rpc'", name="rpc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rpc)
    with pytest.raises(RuntimeError, match="session module 'rpc'.*Offline archive verification"):
        _load_transport_call()
