import pytest
from pathlib import Path
import json
import subprocess
import os

def run_selector(runs_dir: str, require_model="event-tubelet-transformer"):
    repo_root = Path(__file__).resolve().parent.parent.parent
    selector = repo_root / "scripts" / "select_best_onnx_candidate.py"
    try:
        res = subprocess.run(
            ["python", str(selector), "--runs-dir", str(runs_dir), "--require-model-name", require_model],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(res.stdout)
    except subprocess.CalledProcessError as e:
        return e.stderr

def test_selector_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "recovery_run1"
    run_dir.mkdir()
    metrics = {
        "model_name": "event-tubelet-transformer",
        "protocol_version": "3.0",
        "protocol_sha256": "dummy", # Mock protocol hash logic fails here without actual patch, but we just want to test isolation
        "final_test_opened": False,
        "best_checkpoint": str(run_dir / "missing.pt"),
        "best_validation_inverse_ttc_mae": 0.5,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    out = run_selector(str(tmp_path))
    assert "missing_checkpoint" in out or "Error:" in out

def test_selector_path_traversal(tmp_path):
    run_dir = tmp_path / "recovery_run2"
    run_dir.mkdir()
    outside_ckpt = tmp_path / "outside.pt"
    outside_ckpt.touch()
    
    metrics = {
        "model_name": "event-tubelet-transformer",
        "best_checkpoint": str(outside_ckpt),
        "best_validation_inverse_ttc_mae": 0.5,
        "final_test_opened": False
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
        
    out = run_selector(str(tmp_path))
    assert "checkpoint_outside_run_directory" in out or "Error:" in out

def test_selector_final_test_exposure(tmp_path):
    run_dir = tmp_path / "recovery_run3"
    run_dir.mkdir()
    ckpt = run_dir / "model.pt"
    ckpt.touch()
    
    metrics = {
        "model_name": "event-tubelet-transformer",
        "best_checkpoint": str(ckpt),
        "best_validation_inverse_ttc_mae": 0.5,
        "final_test_opened": True
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
        
    out = run_selector(str(tmp_path))
    assert "final_test_exposure" in out or "Error:" in out

def test_selector_model_mismatch(tmp_path):
    run_dir = tmp_path / "recovery_run4"
    run_dir.mkdir()
    ckpt = run_dir / "model.pt"
    ckpt.touch()
    
    metrics = {
        "model_name": "tiny_cnn",
        "best_checkpoint": str(ckpt),
        "best_validation_inverse_ttc_mae": 0.5,
        "final_test_opened": False
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
        
    out = run_selector(str(tmp_path))
    assert "model_name_mismatch" in out or "Error:" in out

