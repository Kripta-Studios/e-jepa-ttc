"""The R4 authority fix changes only the exact scientific proof binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import runpod_rgb_durable_phase_worker_v4 as wrapper


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority(tmp_path: Path) -> Path:
    archive_sha = "a" * 64
    snapshot = tmp_path / f"rgb_individual_r4_{archive_sha[:16]}"
    manifest, local = snapshot / "MANIFEST.json", snapshot / "LOCAL_VERIFICATION.json"
    proof = tmp_path / "scientific_proofs/R4_COMPLETE_REUSE.json"
    write(manifest, {"schema": "ejepa_cloud_rgb_complete_archive_v1"})
    write(local, {"status": "PASS"})
    write(
        proof,
        {
            "schema": "ejepa_local_rgb_scientific_reusability_v1",
            "status": "PASS",
            "archive_sha256": archive_sha,
            "selected_root_path_maps_verified": True,
            "rotations": {
                "4": {
                    "endpoints": {
                        "PAIR_R": {
                            "original_verify_scientific_admission": "PASS",
                            "admission_sha256": "b" * 64,
                            "checkpoint_sha256": "c" * 64,
                        },
                        "R_CTX": {
                            "original_verify_scientific_admission": "PASS",
                            "admission_sha256": "d" * 64,
                            "checkpoint_sha256": "e" * 64,
                        },
                    }
                }
            },
        },
    )
    path = tmp_path / "RAW_R4_EVICTION_ELIGIBILITY.json"
    write(
        path,
        {
            "schema": "ejepa_rgb_raw_rotation_eviction_eligibility_v1",
            "status": "PASS",
            "rotation": 4,
            "scope": "RAW_R4_REPRODUCIBLE",
            "archive_sha256": archive_sha,
            "scientific_genealogy_verified": True,
            "external_uploader_may_use_as_eviction_authority": True,
            "remote_deletion_performed": False,
            "local_deletion_performed": False,
            "archive_manifest_sha256": sha(manifest),
            "local_verification_sha256": sha(local),
            "scientific_reusability_verification_sha256": sha(proof),
        },
    )
    return path


def test_exact_r4_proof_is_admitted_without_legacy_alias(tmp_path: Path) -> None:
    module = wrapper.install_v4()
    path = authority(tmp_path)
    assert not (tmp_path / "scientific_proofs/RGB_COMPLETE_REUSE.json").exists()
    assert module.validate_eviction_authority(path)["status"] == "PASS"


def test_proof_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    module = wrapper.install_v4()
    path = authority(tmp_path)
    proof = tmp_path / "scientific_proofs/R4_COMPLETE_REUSE.json"
    proof.write_text(proof.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or inconsistent"):
        module.validate_eviction_authority(path)


def test_config_binds_authority_fix_and_preserves_state_paths() -> None:
    module = wrapper.install_v4()
    root = Path(__file__).resolve().parents[2]
    config = module.load_config(
        root / "artifacts/runpod_migration_20261026/RGB_DURABLE_PHASE_WORKER_CONFIG_V3.json"
    )
    argv = config["return_coordinator_argv"]
    assert argv[argv.index("--config") + 1].endswith(
        "RGB_RETURN_COORDINATOR_INDIVIDUAL_R4_AUTHORITY_FIX_CONFIG.json"
    )
    assert config["return_coordinator_config_sha256"] == wrapper.AUTHORITY_FIX_CONFIG_SHA256
    assert config["state_path"].endswith("RGB_PHASE_WORKER_STATE.json")
