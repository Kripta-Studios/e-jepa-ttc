import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

@pytest.fixture
def clean_cache(tmp_path: Path):
    npz_path = tmp_path / "cache.npz"
    sidecar_path = tmp_path / "cache.summary.json"

    # Create fake clean cache
    x = np.random.rand(10, 21, 90, 160).astype(np.float32)
    # Ensure no all-zero channel across all samples to pass test 16
    x[:, 0, :, :] += 1.0
    seqs = np.array(["seq1"] * 5 + ["seq2"] * 5)
    splits = np.array(["train"] * 5 + ["validation"] * 5)

    np.savez(npz_path, x=x, sequence_id=seqs, split=splits)

    # Compute actual SHA256 for the sidecar
    sha256_hash = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_sha = sha256_hash.hexdigest()

    sidecar = {
        "format_version": 2,
        "sha256": actual_sha,
        "total_samples": 10,
        "normalization": {
            "enabled": True,
            "strategy": "non_centered_occupied_p95_scale",
            "source_split": "train"
        }
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    return tmp_path


def run_audit(cache_dir, mode="exhaustive"):
    npz_path = cache_dir / "cache.npz"
    out_path = cache_dir / "audit.json"
    cmd = [
        "uv", "run", "python", "scripts/audit_cache.py",
        "--npz-path", str(npz_path),
        "--output", str(out_path),
        "--mode", mode
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if out_path.exists():
        with open(out_path) as f:
            out = json.load(f)
    else:
        out = {}
    return res.returncode, out


def _corrupt_sidecar(cache_dir, modifier_func):
    sidecar_path = cache_dir / "cache.summary.json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    modifier_func(sidecar)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)


def _corrupt_npz(cache_dir, modifier_func):
    npz_path = cache_dir / "cache.npz"
    data = dict(np.load(npz_path))
    modifier_func(data)
    np.savez(npz_path, **data)
    
    # Update sidecar SHA to match the new corrupted NPZ so it doesn't fail on SHA first
    sha256_hash = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_sha = sha256_hash.hexdigest()
    _corrupt_sidecar(cache_dir, lambda s: s.update({"sha256": actual_sha}))


def test_audit_clean_cache(clean_cache):
    code, out = run_audit(clean_cache)
    print("OUT:", out)
    assert code == 0
    assert out["status"] == "passed"


def test_sidecar_sha_mismatch(clean_cache):
    _corrupt_sidecar(clean_cache, lambda s: s.update({"sha256": "bad_sha"}))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("SHA-256 does not match" in e for e in out["failures"])


def test_missing_declared_sha(clean_cache):
    _corrupt_sidecar(clean_cache, lambda s: s.pop("sha256", None))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("SHA-256 does not match" in e for e in out["failures"])


def test_wrong_cache_format_version(clean_cache):
    _corrupt_sidecar(clean_cache, lambda s: s.update({"format_version": 99}))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Invalid cache format version" in e for e in out["failures"])


def test_missing_required_array(clean_cache):
    _corrupt_npz(clean_cache, lambda d: d.pop("split", None))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Missing required array: split" in e for e in out["failures"])


def test_inconsistent_sample_axis_lengths(clean_cache):
    def corrupt(d):
        d["split"] = d["split"][:-1]
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Sample axis inconsistency" in e for e in out["failures"])


def test_invalid_dtype(clean_cache):
    def corrupt(d):
        d["x"] = d["x"].astype(str)
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    print("OUT:", out)
    assert code != 0
    assert any("Invalid x dtype" in e for e in out.get("failures", []))


def test_nan_values(clean_cache):
    def corrupt(d):
        d["x"][0, 0, 0, 0] = np.nan
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("NaN" in e for e in out.get("failures", []))


def test_infinity_values(clean_cache):
    def corrupt(d):
        d["x"][0, 0, 0, 0] = np.inf
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Infinity" in e for e in out.get("failures", []))


def test_empty_cache(clean_cache):
    def corrupt(d):
        d["x"] = d["x"][:0]
        d["sequence_id"] = d["sequence_id"][:0]
        d["split"] = d["split"][:0]
    _corrupt_npz(clean_cache, corrupt)
    _corrupt_sidecar(clean_cache, lambda s: s.update({"total_samples": 0}))
    code, out = run_audit(clean_cache)
    print("OUT:", out)
    assert code != 0
    assert any("Empty cache" in e for e in out.get("failures", []))


def test_missing_sidecar(clean_cache):
    (clean_cache / "cache.summary.json").unlink()
    cmd = ["uv", "run", "python", "scripts/audit_cache.py", "--npz-path", str(clean_cache / "cache.npz"), "--output", str(clean_cache / "audit.json")]
    res = subprocess.run(cmd)
    assert res.returncode != 0


def test_malformed_sidecar(clean_cache):
    with open(clean_cache / "cache.summary.json", "w") as f:
        f.write("{bad json")
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Failed to read sidecar" in e for e in out.get("failures", []))


def test_split_overlap(clean_cache):
    def corrupt(d):
        d["split"][4] = "validation" # seq1 now in train and val
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("belongs to multiple splits" in e for e in out.get("failures", []))


def test_unknown_split(clean_cache):
    def corrupt(d):
        d["split"][0] = "unknown_split"
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Unknown split name" in e for e in out.get("failures", []))


def test_normalizer_fitted_with_validation(clean_cache):
    _corrupt_sidecar(clean_cache, lambda s: s["normalization"].update({"source_split": "validation"}))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("fitted with non-train split" in e for e in out.get("failures", []))


def test_non_empty_source_converted_to_zero(clean_cache):
    def corrupt(d):
        d["x"][0] = 0.0
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("converted to zero" in e for e in out.get("failures", []))


def test_zero_polarity_channel(clean_cache):
    def corrupt(d):
        d["x"][:, 1, :, :] = 0.0 # channel 1 is all zero across all samples
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("broken all-zero channel" in e for e in out.get("failures", []))


def test_missing_navigation_mask(clean_cache):
    def corrupt(d):
        d["navigation"] = np.zeros((9, 4)) # 9 samples instead of 10
    _corrupt_npz(clean_cache, corrupt)
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Navigation sample axis inconsistency" in e for e in out.get("failures", []))


def test_sidecar_belonging_to_another_cache(clean_cache):
    _corrupt_sidecar(clean_cache, lambda s: s.update({"total_samples": 99}))
    code, out = run_audit(clean_cache)
    assert code != 0
    assert any("Declared sample count 99 != actual" in e for e in out.get("failures", []))


def test_nondeterministic_sampled_audit(clean_cache):
    code1, out1 = run_audit(clean_cache, mode="sampled")
    code2, out2 = run_audit(clean_cache, mode="sampled")
    assert code1 == 0 and code2 == 0
    assert out1["checks"]["sampled_indices"] == out2["checks"]["sampled_indices"]
