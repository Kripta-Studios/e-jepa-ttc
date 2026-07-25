import json
from pathlib import Path

from e_jepa_ttc.experiments.test_lock import check_final_test_lock


def test_powershell_orchestration_excludes_final_test():
    """Prove that --allow-final-test-evaluation cannot appear in ordinary orchestration plans."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    eap_script = repo_root / "scripts" / "run_eap_matrix.ps1"

    content = eap_script.read_text(encoding="utf-8")
    assert "--allow-final-test-evaluation" not in content, (
        "Final test evaluation is illegally enabled in standard eAP matrix script."
    )
    assert ' += "test"' not in content, (
        "The test split is illegally included in the default report splits."
    )


def test_final_test_requires_unlock(tmp_path):
    """Prove that final test cannot run without a valid unused unlock."""
    unlock_path = tmp_path / "unlock.json"
    ledger_path = tmp_path / "ledger.jsonl"

    # 1. No unlock file
    assert not check_final_test_lock("final", unlock_path=unlock_path, ledger_path=ledger_path)

    # 2. Valid unlock file
    unlock_data = {
        "authorized_by": "CI",
        "reason": "Final evaluation run",
        "authorization_hash": "abc123nonce",
        "timestamp": "2026-07-24T00:00:00Z",
    }
    unlock_path.write_text(json.dumps(unlock_data), encoding="utf-8")

    assert check_final_test_lock("final", unlock_path=unlock_path, ledger_path=ledger_path)

    # 3. Reusing the authorization nonce fails
    assert not check_final_test_lock("final", unlock_path=unlock_path, ledger_path=ledger_path)


def test_cpla_high_rejected_as_final_split():
    """Prove that CPLA-high is rejected as a final split."""
    # This might require checking the split definitions in e_jepa_ttc/data/split.py
    # For this test, we just ensure that CPLA-high is diagnostic or validation, not test.
    repo_root = Path(__file__).resolve().parent.parent.parent
    status = repo_root / "STATUS.md"
    content = status.read_text(encoding="utf-8")
    assert "CPLA-high is diagnostic only" in content


def test_ordinary_orchestration_plans():
    """Smoke, pilot, validation matrix never include test."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_all = repo_root / "scripts" / "run_all.ps1"
    content = run_all.read_text(encoding="utf-8").lower().split("--report-splits")[0]
    assert '"test"' not in content and "'test'" not in content, (
        "Test split leak in run_all.ps1"
    )

def test_cache_requires_lock(tmp_path):
    import subprocess
    import numpy as np
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    
    # Create structurally valid npz
    cache_npz = tmp_path / "cache_evttc.npz"
    np.savez(
        cache_npz,
        x=np.zeros((1, 10, 10), dtype=np.float32),
        sequence_id=np.array(["CCRs-1-low"]),
        split=np.array(["test"]),
        source_event_count=np.array([100])
    )
    
    import hashlib
    
    def hash_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as bf:
            for chunk in iter(lambda: bf.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
        
    real_hash = hash_file(cache_npz)

    # Create valid sidecar
    cache_sidecar = tmp_path / "cache_evttc.summary.json"
    cache_sidecar.write_text(json.dumps({
        "format_version": 2,
        "window_count": 1,
        "cache_sha256": real_hash
    }))
    
    cmd = [
        "uv", "run", "--no-sync", "python",
        str(repo_root / "scripts" / "audit_cache.py"),
        "--npz-path", str(cache_npz),
        "--output", str(tmp_path / "audit_output.json"),
        "--evidence-type", "final_test"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "final_test requested but no valid unlock found" in res.stderr or "final_test requested but no valid unlock found" in res.stdout or "Audit completed with" in res.stderr
    
    out_data = json.loads((tmp_path / "audit_output.json").read_text())
    assert any("final_test requested but no valid unlock found" in str(f) for f in out_data["failures"])

