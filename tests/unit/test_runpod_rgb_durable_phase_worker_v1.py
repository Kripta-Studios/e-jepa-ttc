"""Durability tests for the remaining RGB phase worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import runpod_rgb_durable_phase_worker_v1 as worker


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def record(relative: str = "data/train/a/rgb_shards/rgb-00000.tar") -> dict[str, object]:
    return {
        "relative": relative,
        "sparse_tar_bytes": 3,
        "sparse_tar_sha256": hashlib.sha256(b"rgb").hexdigest(),
    }


def test_upload_completion_requires_exact_manifest_ledger(tmp_path: Path) -> None:
    expected = {str(record()["relative"]): record()}
    status = tmp_path / "status.json"
    value = {
        "schema": "ejepa_rgb_sparse_parallel_uploader_v2",
        "plan_sha256": "a" * 64,
        "active": [],
        "completed": {
            str(record()["relative"]): {
                "bytes": 3,
                "sha256": hashlib.sha256(b"rgb").hexdigest(),
                "marker_sha256": "b" * 64,
            }
        },
    }
    write_json(status, value)
    assert worker.upload_complete(status, expected, "a" * 64)
    value["completed"][str(record()["relative"])]["sha256"] = "c" * 64
    write_json(status, value)
    with pytest.raises(ValueError, match="identity differs"):
        worker.upload_complete(status, expected, "a" * 64)


def test_live_exact_process_is_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = object.__new__(worker.Worker)
    instance.state = {
        "processes": {"upload": {"pid": 12, "argv": ["python", "upload.py"]}},
        "attempts": {},
    }
    instance.config = {"max_launch_attempts": 12}
    launched: list[object] = []
    instance.popen = lambda *_args, **_kwargs: launched.append(object())
    monkeypatch.setattr(
        worker, "process_matches", lambda pid, argv: (pid, argv) == (12, ["python", "upload.py"])
    )
    instance.launch("upload", ["python", "upload.py"])
    assert launched == []


def marker(payload: bytes) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "ejepa_rgb_sparse_tar_receipt_v1",
        "sparse_tar_sha256": hashlib.sha256(payload).hexdigest(),
    }
    value["receipt_sha256"] = worker.canonical(value)
    return value


def eviction_payload(relative: str, payload: bytes) -> dict[str, object]:
    receipt = marker(payload)
    return {
        "authority_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "records": [
            {
                "relative": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "receipt_sha256": receipt["receipt_sha256"],
            }
        ],
    }


def run_eviction(root: Path, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-c", worker.REMOTE_EVICTION, str(root), json.dumps(payload)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_eviction_exact_retains_marker_and_resumes(tmp_path: Path) -> None:
    relative = "data/train/a/rgb_shards/rgb-00000.tar"
    tar = tmp_path.joinpath(*relative.split("/"))
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b"rgb")
    sidecar = tar.with_name(tar.name + ".json")
    write_json(sidecar, marker(b"rgb"))
    payload = eviction_payload(relative, b"rgb")
    first = run_eviction(tmp_path, payload)
    assert first.returncode == 0, first.stderr
    assert not tar.exists() and sidecar.is_file()
    second = run_eviction(tmp_path, payload)
    assert second.returncode == 0, second.stderr


def test_eviction_missing_without_intent_fails_closed(tmp_path: Path) -> None:
    relative = "data/train/a/rgb_shards/rgb-00000.tar"
    sidecar = tmp_path.joinpath(*(relative + ".json").split("/"))
    write_json(sidecar, marker(b"rgb"))
    result = run_eviction(tmp_path, eviction_payload(relative, b"rgb"))
    assert result.returncode != 0
    assert not (tmp_path / "R4_RAW_EVICTION_INTENT.json").exists()


def test_eviction_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    tar = outside / "train/a/rgb_shards/rgb-00000.tar"
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b"rgb")
    write_json(tar.with_name(tar.name + ".json"), marker(b"rgb"))
    try:
        (tmp_path / "data").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    relative = "data/train/a/rgb_shards/rgb-00000.tar"
    result = run_eviction(tmp_path, eviction_payload(relative, b"rgb"))
    assert result.returncode != 0
    assert tar.is_file()


def test_bounded_restart_records_pid(tmp_path: Path) -> None:
    instance = object.__new__(worker.Worker)
    instance.state = {"processes": {}, "attempts": {}}
    instance.config = {
        "max_launch_attempts": 1,
        "log_root": str(tmp_path),
        "repository": str(tmp_path),
    }
    instance.popen = lambda *_args, **_kwargs: SimpleNamespace(pid=44)
    instance.publish = lambda detail: None
    instance.launch("upload", ["python", "upload.py"])
    assert instance.state["processes"]["upload"]["pid"] == 44
    instance.state["processes"] = {}
    with pytest.raises(RuntimeError, match="bounded"):
        instance.launch("upload", ["python", "upload.py"])


def test_eviction_config_rejects_different_remote_root() -> None:
    with pytest.raises(ValueError, match="remote root"):
        worker.validate_eviction_config({"remote_root": "/workspace/other", "ssh_argv_prefix": []})
