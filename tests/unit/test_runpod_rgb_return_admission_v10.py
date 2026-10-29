from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from operational.runpod_migration.rgb_return_admission_v10 import (
    _verified_manifest_v10,
    admit_local_rgb_return,
)
from operational.runpod_migration.rgb_return_archive import ARCHIVE_SCHEMA, PILOT_SHA256, sha256


def _archive(path: Path, *, r2_physical: int = 2) -> str:
    payload = b"global-v10"
    files = {
        "roots/r2_downstream/output.bin": {
            "source_path": "/workspace/ejepa_technical/r2_rgb_downstream_v1/output.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    media = {
        f"R{rotation}_{role}": {
            "plan_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "physical_root": "/opt/ejepa_rgb_frames",
            "tar_payloads_included": False,
            "tar_receipts": [],
        }
        for rotation in (2, 3, 4)
        for role in ("refiner", "generation")
    }
    for scope, physical, role, mode in (
        ("R2_refiner", 3, "generation", "REUSE"),
        ("R2_generation", r2_physical, "generation", "DIRECT"),
        ("R4_refiner", 4, "refiner", "DIRECT"),
        ("R4_generation", 3, "refiner", "REUSE"),
    ):
        media[scope] = {
            "admission_schema": "ejepa_rgb_compositional_scope_admission_v8",
            "composition_mode": mode,
            "physical_rotation": physical,
            "physical_role": role,
            "physical_root": "/workspace/ejepa_rgb_frames",
            "archived_genealogy": {},
            "tar_payloads_included": False,
            "tar_receipts": [],
        }
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "status": "READY",
        "rotations": [2, 3, 4],
        "campaign_complete": True,
        "individual_rotation_return": False,
        "progress_presented_as_endpoint": False,
        "input_bindings": {"RGB_PILOT.tar": {"sha256": PILOT_SHA256}},
        "active_rgb_context_chain": ["V5", "V6", "V7", "V8"],
        "active_rgb_context_chain_by_rotation": {
            "2": ["V5", "V6", "V7", "V8", "V9"],
            "3": ["V5", "V6", "V7", "V9"],
            "4": ["V5", "V6", "V7", "V8"],
        },
        "compositional_v8_rotations": [2, 4],
        "operational_v9_rotations": [2, 3],
        "global_context_closure": "R2_V8_V9__R3_V7_V9__R4_V8",
        "sparse_rgb_tar_payloads_included": False,
        "endpoints": {
            f"R{rotation}_{fit}": {}
            for rotation in (2, 3, 4)
            for fit in ("R_A5", "R_C2F", "PAIR_R", "R_CTX")
        },
        "rgb_media_scope_evidence": media,
        "content_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": files,
    }
    encoded = (json.dumps(manifest) + "\n").encode()
    with tarfile.open(path, "w") as stream:
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(encoded)
        stream.addfile(info, io.BytesIO(encoded))
        info = tarfile.TarInfo("roots/r2_downstream/output.bin")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    return sha256(path)


def test_v10_windows_manifest_accepts_physical_r2_and_binds_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "global.tar"
    digest = _archive(archive)
    manifest, _ = _verified_manifest_v10(archive, digest, expected_rotations=(2, 3, 4))
    assert manifest["compositional_v8_rotations"] == [2, 4]
    assert manifest["rgb_media_scope_evidence"]["R2_generation"]["physical_rotation"] == 2


def test_v10_windows_manifest_rejects_r2_physical_relabel(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    digest = _archive(archive, r2_physical=3)
    with pytest.raises(ValueError, match="compositional evidence"):
        _verified_manifest_v10(archive, digest, expected_rotations=(2, 3, 4))


def test_v10_full_admission_uses_overlay_without_recursion(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    archive = tmp_path / "global.tar"
    digest = _archive(archive)
    result = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb",
        repository=repository,
    )
    assert result["status"] == "PASS"
    assert result["rotations"] == [2, 3, 4]
    assert result["cleanup"]["status"] == "NOT_EVALUATED"
