import json
import subprocess
from pathlib import Path


def test_onnx_export_rejects_altered_record(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    record_path = tmp_path / "selection.json"
    onnx_out = tmp_path / "model.onnx"

    # Create fake files so hash checks actually run
    chk_path = tmp_path / "model.pt"
    chk_path.write_text("fake_chk")
    cache_path = tmp_path / "cache.npz"
    cache_path.write_text("fake_cache")

    # Write a record with deliberate hash mismatch
    record = {
        "checkpoint_path": str(chk_path),
        "checkpoint_sha256": "wronghash",
        "cache_path": str(cache_path),
        "cache_sha256": "wronghash",
        "model_config_path": "unknown",
        "model_config_sha256": "unknown",
        "protocol_hash": "unknown",
        "code_commit": "unknown",
    }

    record_path.write_text(json.dumps(record))

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(repo_root / "scripts" / "export_onnx.py"),
        "--selection-record",
        str(record_path),
        "--output",
        str(onnx_out),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Checkpoint hash mismatch" in res.stderr or "Cache hash mismatch" in res.stderr


def test_onnx_export_rejects_final_test_checkpoint(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    record_path = tmp_path / "selection.json"
    onnx_out = tmp_path / "model.onnx"

    # Create fake files with correct hashes
    import hashlib

    def get_hash(p):
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    chk_path = tmp_path / "model.pt"
    chk_path.write_text("fake_chk")
    cache_path = tmp_path / "cache.npz"
    cache_path.write_text("fake_cache")

    # Simulate final_test_opened = True
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"final_test_opened": True}))

    record = {
        "checkpoint_path": str(chk_path),
        "checkpoint_sha256": get_hash(chk_path),
        "cache_path": str(cache_path),
        "cache_sha256": get_hash(cache_path),
        "model_config_path": "unknown",
        "model_config_sha256": "unknown",
        "protocol_hash": "unknown",
        "code_commit": "unknown",
    }

    record_path.write_text(json.dumps(record))

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(repo_root / "scripts" / "export_onnx.py"),
        "--selection-record",
        str(record_path),
        "--output",
        str(onnx_out),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Selected checkpoint was exposed to final test" in res.stderr
