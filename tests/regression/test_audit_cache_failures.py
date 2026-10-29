import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def test_audit_cache_fails_on_malformed_sidecar(tmp_path):
    npz_path = tmp_path / "cache.npz"
    np.savez(npz_path, x=np.zeros((1,)), sequence_id=np.array(["seq1"]), split=np.array(["train"]))

    sha256_hash = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_sha = sha256_hash.hexdigest()

    sidecar = tmp_path / "cache.summary.json"
    sidecar.write_text(json.dumps({"bad_json": True, "sha256": actual_sha}))

    audit_out = tmp_path / "audit.json"
    repo_root = Path(__file__).resolve().parent.parent.parent
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(repo_root / "scripts" / "audit_cache.py"),
        "--npz-path",
        str(npz_path),
        "--output",
        str(audit_out),
        "--evidence-type",
        "validation_matrix",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0

    if audit_out.exists():
        data = json.loads(audit_out.read_text())
        assert data["status"] == "failed"
        assert len(data["failures"]) > 0


def test_audit_cache_fails_test_split(tmp_path):
    npz_path = tmp_path / "cache.npz"
    # Using 'test' split which is not allowed in validation_matrix
    np.savez(
        npz_path,
        x=np.zeros((1, 5, 2, 2)),
        sequence_id=np.array(["seq1"]),
        split=np.array(["test"]),
        event_count=np.array([10]),
    )
    sha256_hash = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_sha = sha256_hash.hexdigest()

    sidecar = tmp_path / "cache.summary.json"
    sidecar.write_text(json.dumps({"format_version": 2, "sha256": actual_sha, "total_samples": 1}))

    audit_out = tmp_path / "audit.json"
    repo_root = Path(__file__).resolve().parent.parent.parent
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(repo_root / "scripts" / "audit_cache.py"),
        "--npz-path",
        str(npz_path),
        "--output",
        str(audit_out),
        "--evidence-type",
        "validation_matrix",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0

    data = json.loads(audit_out.read_text())
    assert data["status"] == "failed"
    assert any("Final test split is present" in f for f in data["failures"]), (
        f"Failures were: {data['failures']}"
    )


def test_audit_cache_reads_current_builder_metadata_without_float16_overflow(tmp_path):
    npz_path = tmp_path / "cache.npz"
    np.savez(
        npz_path,
        x=np.full((2, 2, 2, 2), 60_000, dtype=np.float16),
        sequence_id=np.array(["seq1", "seq1"]),
        split=np.array(["train", "train"]),
        event_count=np.array([10, 10]),
        cache_format_version=np.array(2),
        normalize=np.array(False),
        normalization=np.array("none"),
    )
    actual_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    sidecar = tmp_path / "cache.summary.json"
    sidecar.write_text(
        json.dumps(
            {
                "cache_sha256": actual_sha,
                "window_count": 2,
            }
        )
    )
    audit_out = tmp_path / "audit.json"
    repo_root = Path(__file__).resolve().parent.parent.parent
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(repo_root / "scripts" / "audit_cache.py"),
        "--npz-path",
        str(npz_path),
        "--output",
        str(audit_out),
        "--evidence-type",
        "validation_matrix",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "overflow" not in result.stderr.lower()
    audit = json.loads(audit_out.read_text())
    assert audit["cache_format_version"] == 2
    assert audit["normalization"] == "none"
    assert audit["normalizer_source_split"] == "not_applicable"
    assert audit["normalizer_origins_verified"] is True
