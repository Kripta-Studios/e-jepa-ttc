import json
import subprocess
from pathlib import Path
import torch
import numpy as np


def test_onnx_export_rejects_altered_record(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    record_path = tmp_path / "selection.json"
    onnx_out = tmp_path / "model.onnx"

    # Create structurally valid files so hash checks actually run
    chk_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": {}, "resolved_model_config": {}}, chk_path)
    
    cache_path = tmp_path / "cache.npz"
    np.savez(cache_path, x=np.zeros((1, 1), dtype=np.float32))

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

    # Create structurally valid files with correct hashes
    import hashlib

    def get_hash(p):
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    chk_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": {}, "resolved_model_config": {}}, chk_path)
    
    cache_path = tmp_path / "cache.npz"
    np.savez(cache_path, x=np.zeros((1, 1), dtype=np.float32))

    # Simulate final_test_opened = True
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"final_test_opened": True}))

    # Realistic fallback values for required fields
    empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    record = {
        "checkpoint_path": str(chk_path),
        "checkpoint_sha256": get_hash(chk_path),
        "cache_path": str(cache_path),
        "cache_sha256": get_hash(cache_path),
        "model_config_path": "configs/model/tiny_cnn.yaml",
        "model_config_sha256": empty_sha256,
        "protocol_hash": empty_sha256,
        "code_commit": "d88f571"
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
