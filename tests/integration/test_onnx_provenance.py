import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity


def test_onnx_export_rejects_altered_record(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    record_path = tmp_path / "selection.json"
    onnx_out = tmp_path / "model.onnx"

    # Create structurally valid files so hash checks actually run
    chk_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": {}, "resolved_model_config": {}}, chk_path)

    cache_path = tmp_path / "cache.npz"
    np.savez(cache_path, x=np.zeros((1, 1), dtype=np.float32))

    protocol_version, protocol_sha256 = get_current_protocol_identity()
    # Write a signed record with a deliberate physical hash mismatch.
    record = {
        "artifact_type": "onnx_candidate_v3",
        "schema_version": "3.0",
        "checkpoint_path": str(chk_path),
        "checkpoint_sha256": "wronghash",
        "cache_path": str(cache_path),
        "cache_sha256": "wronghash",
        "model_config_path": "unknown",
        "model_config_sha256": "unknown",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "code_commit": "unknown",
        "selection_split": "validation",
        "evidence_type": "validation_matrix",
        "created_at": "2026-07-25",
        "artifact_sha256": "unknown",
        "cache_sidecar_sha256": "unknown",
        "model_config": {},
        "split_manifest_sha256": "unknown",
        "normalization_sha256": "unknown",
        "navigation_mode": "disabled",
        "selection_metric": "mae_s",
        "selection_metric_value": 0.0,
        "run_id": "fake",
        "model_name": "fake",
        "seed": 7,
        "label_fraction": 1.0,
        "train_sample_count": 0,
        "validation_sample_count": 0,
        "final_test_opened": False,
    }

    record_path.write_text(json.dumps(sign_artifact(record)))

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

    protocol_version, protocol_sha256 = get_current_protocol_identity()
    # Realistic fallback values for required fields
    empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    record = {
        "artifact_type": "onnx_candidate_v3",
        "schema_version": "3.0",
        "checkpoint_path": str(chk_path),
        "checkpoint_sha256": get_hash(chk_path),
        "cache_path": str(cache_path),
        "cache_sha256": get_hash(cache_path),
        "model_config_path": "configs/model/tiny_cnn.yaml",
        "model_config_sha256": empty_sha256,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "code_commit": "d88f571",
        "selection_split": "validation",
        "evidence_type": "validation_matrix",
        "created_at": "2026-07-25",
        "artifact_sha256": empty_sha256,
        "cache_sidecar_sha256": empty_sha256,
        "model_config": {},
        "split_manifest_sha256": empty_sha256,
        "normalization_sha256": empty_sha256,
        "navigation_mode": "disabled",
        "selection_metric": "mae_s",
        "selection_metric_value": 0.0,
        "run_id": "fake",
        "model_name": "fake",
        "seed": 7,
        "label_fraction": 1.0,
        "train_sample_count": 0,
        "validation_sample_count": 0,
        "final_test_opened": False,
    }

    record_path.write_text(json.dumps(sign_artifact(record)))

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
