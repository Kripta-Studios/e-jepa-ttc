import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

@pytest.fixture
def clean_cache(tmp_path: Path):
    npz_path = tmp_path / "cache.npz"
    sidecar_path = tmp_path / "cache.summary.json"
    
    # Create fake clean cache
    x = np.random.rand(10, 21, 90, 160).astype(np.float16)
    seqs = np.array(["seq1"] * 5 + ["seq2"] * 5)
    splits = np.array(["train"] * 5 + ["val"] * 5)
    
    np.savez(npz_path, x=x, sequence_id=seqs, split=splits)
    
    sidecar = {
        "shape": [10, 21, 90, 160],
        "normalize": True
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)
        
    return tmp_path

def test_audit_clean_cache(clean_cache):
    npz_path = clean_cache / "cache.npz"
    out_path = clean_cache / "audit.json"
    
    res = subprocess.run(["uv", "run", "python", "scripts/audit_cache.py", "--npz-path", str(npz_path), "--output", str(out_path)])
    assert res.returncode == 0
    
    with open(out_path) as f:
        data = json.load(f)
    assert data["valid_nan_inf"] is True
    assert data["valid_tensor_shapes"] is True
    assert data["split_disjointness"] is True
    assert len(data["errors"]) == 0


def test_audit_nan_cache(clean_cache):
    npz_path = clean_cache / "cache.npz"
    out_path = clean_cache / "audit.json"
    
    # Corrupt with NaN
    data = dict(np.load(npz_path))
    data["x"][0, 0, 0, 0] = np.nan
    np.savez(npz_path, **data)
    
    res = subprocess.run(["uv", "run", "python", "scripts/audit_cache.py", "--npz-path", str(npz_path), "--output", str(out_path)])
    assert res.returncode != 0
    
    with open(out_path) as f:
        out = json.load(f)
    assert out["valid_nan_inf"] is False
    assert any("NaN" in e for e in out["errors"])


def test_audit_split_leakage(clean_cache):
    npz_path = clean_cache / "cache.npz"
    out_path = clean_cache / "audit.json"
    
    # Corrupt splits
    data = dict(np.load(npz_path))
    data["split"][4] = "val"  # seq1 now in train AND val
    np.savez(npz_path, **data)
    
    res = subprocess.run(["uv", "run", "python", "scripts/audit_cache.py", "--npz-path", str(npz_path), "--output", str(out_path)])
    assert res.returncode != 0
    
    with open(out_path) as f:
        out = json.load(f)
    assert out["split_disjointness"] is False
    assert any("multiple splits" in e for e in out["errors"])

def test_audit_shape_mismatch(clean_cache):
    npz_path = clean_cache / "cache.npz"
    out_path = clean_cache / "audit.json"
    
    # Corrupt sidecar expected shape
    sidecar_path = clean_cache / "cache.summary.json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    sidecar["shape"] = [10, 21, 90, 150] # wrong width
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)
        
    res = subprocess.run(["uv", "run", "python", "scripts/audit_cache.py", "--npz-path", str(npz_path), "--output", str(out_path)])
    assert res.returncode != 0
    
    with open(out_path) as f:
        out = json.load(f)
    assert out["valid_tensor_shapes"] is False
    assert any("Shape mismatch" in e for e in out["errors"])
