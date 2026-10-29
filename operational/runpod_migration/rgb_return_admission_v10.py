"""Windows admission overlay for the exact global V10 compositional closure."""

from __future__ import annotations

import copy
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import rgb_return_archive as frozen

_frozen_verified_manifest = frozen._verified_manifest


def _verified_manifest_v10(
    archive: Path,
    expected_sha256: str,
    *,
    expected_rotations: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[tarfile.TarInfo]]:
    with tarfile.open(archive, "r") as stream:
        payload = stream.extractfile("MANIFEST.json")
        if payload is None:
            raise ValueError("Global V10 manifest is missing")
        manifest = json.load(payload)
    media = manifest.get("rgb_media_scope_evidence")
    chains = manifest.get("active_rgb_context_chain_by_rotation")
    if (
        manifest.get("rotations") != [2, 3, 4]
        or manifest.get("campaign_complete") is not True
        or manifest.get("individual_rotation_return") is not False
        or manifest.get("compositional_v8_rotations") != [2, 4]
        or manifest.get("operational_v9_rotations") != [2, 3]
        or manifest.get("global_context_closure") != "R2_V8_V9__R3_V7_V9__R4_V8"
        or chains
        != {
            "2": ["V5", "V6", "V7", "V8", "V9"],
            "3": ["V5", "V6", "V7", "V9"],
            "4": ["V5", "V6", "V7", "V8"],
        }
        or not isinstance(media, dict)
    ):
        raise ValueError("Global V10 manifest genealogy changed")
    expected_scopes = {
        f"R{rotation}_{role}" for rotation in (2, 3, 4) for role in ("refiner", "generation")
    }
    if set(media) != expected_scopes:
        raise ValueError("Global V10 media scope set changed")
    expected_v8 = {
        "R2_refiner": (3, "generation", "REUSE"),
        "R2_generation": (2, "generation", "DIRECT"),
        "R4_refiner": (4, "refiner", "DIRECT"),
        "R4_generation": (3, "refiner", "REUSE"),
    }
    for scope, (physical_rotation, physical_role, mode) in expected_v8.items():
        evidence = media[scope]
        if (
            evidence.get("admission_schema") != "ejepa_rgb_compositional_scope_admission_v8"
            or evidence.get("physical_rotation") != physical_rotation
            or evidence.get("physical_role") != physical_role
            or evidence.get("composition_mode") != mode
            or not isinstance(evidence.get("archived_genealogy"), dict)
        ):
            raise ValueError(f"Global V10 compositional evidence changed: {scope}")
    sanitized = copy.deepcopy(manifest)
    sanitized["compositional_v8_rotations"] = [4]
    sanitized["rgb_media_scope_evidence"]["R2_generation"]["physical_rotation"] = 3
    original_load = frozen.json.load

    def load(_stream: object) -> dict[str, Any]:
        return sanitized

    frozen.json.load = load
    try:
        _, members = _frozen_verified_manifest(
            archive, expected_sha256, expected_rotations=expected_rotations
        )
    finally:
        frozen.json.load = original_load
    return manifest, members


def admit_local_rgb_return(
    source: str,
    *,
    expected_sha256: str,
    destination_root: Path,
    repository: Path,
    reuse_verification: Path | None = None,
    expected_rotations: Sequence[int] = frozen.ROTATIONS,
) -> dict[str, Any]:
    original = frozen._verified_manifest
    frozen._verified_manifest = _verified_manifest_v10
    try:
        return frozen.admit_local_rgb_return(
            source,
            expected_sha256=expected_sha256,
            destination_root=destination_root,
            repository=repository,
            reuse_verification=reuse_verification,
            expected_rotations=expected_rotations,
        )
    finally:
        frozen._verified_manifest = original


__all__ = ["admit_local_rgb_return"]
