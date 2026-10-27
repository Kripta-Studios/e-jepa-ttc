from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from operational.runpod_migration.event_handoff import _portable_source_identity
from operational.runpod_migration.immutable_seal import create, verify
from scripts.runpod_event_transfer_v1 import REMOTE_RECEIVER


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_os_source_equivalence_ignores_only_physical_paths() -> None:
    windows = {
        "schema": "fold",
        "rotation": 2,
        "identity_sha256": "windows-digest",
        "storage_source_identity": {
            "H": {
                "role": "H",
                "role_manifest_path": "C:\\source\\H_MANIFEST.json",
                "event_source_root": "E:\\events",
                "rows_sha256": "abc",
                "identity_sha256": "windows-role-digest",
            }
        },
    }
    linux = json.loads(json.dumps(windows))
    linux["identity_sha256"] = "linux-digest"
    linux["storage_source_identity"]["H"].update(
        role_manifest_path="/opt/bundle/H_MANIFEST.json",
        event_source_root="/opt/bundle/events",
        identity_sha256="linux-role-digest",
    )
    assert _portable_source_identity(windows) == _portable_source_identity(linux)
    linux["storage_source_identity"]["H"]["rows_sha256"] = "changed"
    assert _portable_source_identity(windows) != _portable_source_identity(linux)


def test_immutable_seal_is_fast_and_fail_closed(tmp_path: Path) -> None:
    logical, stage = tmp_path / "logical", tmp_path / "stage"
    logical.mkdir()
    stage.mkdir()
    payload = stage / "data.bin"
    payload.write_bytes(b"event-bytes")
    payload.chmod(0o444)
    manifest = {
        "files": [
            {
                "relative": "data.bin",
                "bytes": payload.stat().st_size,
                "sha256": _sha(payload),
            }
        ]
    }
    manifest_path = stage / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    content_rows = [
        {"relative": "data.bin", "bytes": payload.stat().st_size, "sha256": _sha(payload)}
    ]
    content = hashlib.sha256(
        json.dumps(content_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    readiness = tmp_path / "READY.json"
    readiness.write_text(
        json.dumps(
            {
                "status": "READY",
                "stage_root": str(stage.resolve()),
                "bundle_manifest_sha256": _sha(manifest_path),
                "content_identity_sha256": content,
            }
        ),
        encoding="utf-8",
    )
    seal = tmp_path / "SEAL.json"
    create(stage, readiness, seal)
    assert verify(stage, seal) == manifest
    payload.chmod(0o644)
    payload.write_bytes(b"tampered!!!")
    with pytest.raises(ValueError, match="changed"):
        verify(stage, seal)


def _line(process: subprocess.Popen[bytes]) -> dict:
    assert process.stdout is not None
    return json.loads(process.stdout.readline())


def test_event_transfer_receiver_resumes_verifies_and_seals(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    root.mkdir()
    payload = b"0123456789abcdef"
    expected = hashlib.sha256(payload).hexdigest()
    command = [sys.executable, "-u", "-c", REMOTE_RECEIVER, str(root)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert process.stdin is not None
    header = {"relative": "cache/shard.bin", "bytes": len(payload), "sha256": expected}
    process.stdin.write((json.dumps(header) + "\n").encode())
    process.stdin.flush()
    assert _line(process)["offset"] == 0
    process.stdin.write(payload)
    process.stdin.flush()
    assert _line(process)["status"] == "OK"
    process.stdin.write((json.dumps(header) + "\n").encode())
    process.stdin.flush()
    assert _line(process)["status"] == "SKIP"
    process.stdin.write(b'{"command":"END"}\n')
    process.stdin.flush()
    assert _line(process)["status"] == "COMPLETE"
    process.stdin.close()
    assert process.wait(timeout=5) == 0
    target = root / "cache/shard.bin"
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o222 == 0
    seal = json.loads(target.with_name("shard.bin.transfer.json").read_text())
    assert seal == {
        "schema": "ejepa_ssh_transfer_seal_v1",
        "bytes": len(payload),
        "sha256": expected,
    }
