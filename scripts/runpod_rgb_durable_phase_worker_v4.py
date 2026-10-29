"""Bind the corrected individual-R4 authority paths around frozen worker V3."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

V3_SHA256 = "609ad505bfc85db94b2666146694944e871643f902fb675b2f72cf10346b61d3"
BASE_CONFIG_SHA256 = "ff3e4d74ca0452daadee0863de13d596c28e4d2c05b6b0112a0c5a763d035ab5"
AUTHORITY_FIX_CONFIG_SHA256 = "5dc9436f393be46487aee972ae1d84ca992aad4a9ec9781d2263eb196c2dbc5d"
PROOF_NAME = "R4_COMPLETE_REUSE.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v4() -> Any:
    source = Path(__file__).resolve()
    v3_path = source.with_name("runpod_rgb_durable_phase_worker_v3.py")
    if digest(v3_path) != V3_SHA256:
        raise ValueError("Frozen durable RGB worker V3 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_rgb_durable_phase_worker_v3_v4", v3_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(v3_path))
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v3()
    original_load = module.load_config

    def load_config(path: Path) -> dict[str, Any]:
        overlay = module.read_json(path.resolve(strict=True))
        if overlay.get("schema") != "ejepa_rgb_durable_phase_worker_config_v3":
            raise ValueError("Durable RGB worker V3 config schema differs")
        base = Path(str(overlay.get("base_config_path"))).resolve(strict=True)
        authority = Path(str(overlay.get("authority_fix_config_path"))).resolve(strict=True)
        if (
            digest(base) != BASE_CONFIG_SHA256
            or overlay.get("base_config_sha256") != BASE_CONFIG_SHA256
        ):
            raise ValueError("Frozen durable RGB worker V2 config differs")
        if (
            digest(authority) != AUTHORITY_FIX_CONFIG_SHA256
            or overlay.get("authority_fix_config_sha256") != AUTHORITY_FIX_CONFIG_SHA256
        ):
            raise ValueError("Individual R4 authority-fix config bytes differ")
        value = original_load(base)
        argv = value["return_coordinator_argv"]
        index = argv.index("--config") + 1
        argv[index] = str(authority)
        value["return_coordinator_config_sha256"] = AUTHORITY_FIX_CONFIG_SHA256
        value["config_authority_fix_overlay_sha256"] = digest(path.resolve(strict=True))
        return value

    def validate_eviction_authority(path: Path) -> dict[str, Any]:
        value = module.read_json(path)
        root = path.parent.resolve(strict=True)
        archive_sha = value.get("archive_sha256")
        if not isinstance(archive_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
            raise ValueError("Individual R4 archive identity is invalid")
        snapshot = root / f"rgb_individual_r4_{archive_sha[:16]}"
        if snapshot.is_symlink():
            raise ValueError("Individual R4 snapshot root cannot be a symlink")
        snapshot = snapshot.resolve(strict=True)
        if snapshot.parent != root:
            raise ValueError("Individual R4 snapshot root escaped")
        manifest = snapshot / "MANIFEST.json"
        local = snapshot / "LOCAL_VERIFICATION.json"
        scientific = root / "scientific_proofs" / PROOF_NAME
        if (
            value.get("schema") != "ejepa_rgb_raw_rotation_eviction_eligibility_v1"
            or value.get("status") != "PASS"
            or value.get("rotation") != 4
            or value.get("scope") != "RAW_R4_REPRODUCIBLE"
            or value.get("scientific_genealogy_verified") is not True
            or value.get("external_uploader_may_use_as_eviction_authority") is not True
            or value.get("remote_deletion_performed") is not False
            or value.get("local_deletion_performed") is not False
            or value.get("archive_manifest_sha256") != module.sha256(manifest)
            or value.get("local_verification_sha256") != module.sha256(local)
            or value.get("scientific_reusability_verification_sha256") != module.sha256(scientific)
        ):
            raise ValueError("Individual R4 eviction authority is absent or inconsistent")
        proof = module.read_json(scientific)
        endpoints = proof.get("rotations", {}).get("4", {}).get("endpoints", {})
        endpoint_pass = isinstance(endpoints, dict) and set(endpoints) == {"PAIR_R", "R_CTX"}
        if endpoint_pass:
            endpoint_pass = all(
                item.get("original_verify_scientific_admission") == "PASS"
                and isinstance(item.get("admission_sha256"), str)
                and isinstance(item.get("checkpoint_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["admission_sha256"])
                and re.fullmatch(r"[0-9a-f]{64}", item["checkpoint_sha256"])
                for item in endpoints.values()
            )
        if (
            proof.get("schema") != "ejepa_local_rgb_scientific_reusability_v1"
            or proof.get("status") != "PASS"
            or proof.get("archive_sha256") != value.get("archive_sha256")
            or proof.get("selected_root_path_maps_verified") is not True
            or set(proof.get("rotations", {})) != {"4"}
            or not endpoint_pass
        ):
            raise ValueError("R4 scientific proof does not authorize eviction")
        return value

    module.load_config = load_config
    module.validate_eviction_authority = validate_eviction_authority
    return module


def main() -> int:
    return int(install_v4().main())


if __name__ == "__main__":
    raise SystemExit(main())
