"""Verified multi-root routing for immutable RGB producer endpoint directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .inventory import digest


def _verify_cloud_endpoint(
    producer_run: Path, admission: dict[str, Any], rotation: int, fit: str
) -> dict[str, str]:
    """Bind a scientific endpoint to its original cloud execution admission."""
    from .inventory import read

    migration_path = producer_run / "MIGRATION_BINDING.json"
    cloud_path = producer_run / "SCIENTIFIC_CLOUD_ADMISSION.json"
    migration, cloud = read(migration_path), read(cloud_path)
    expected_schema = (
        "ejepa_scientific_cloud_admission_c2f_v1"
        if fit == "R_C2F"
        else f"ejepa_scientific_cloud_admission_v{4 if rotation == 4 else 5}"
    )
    migration_sha, cloud_sha = digest(migration_path), digest(cloud_path)
    if (
        migration.get("schema") != "ejepa_linux_technical_migration_binding_v3"
        or migration.get("scientific_files_rewritten") is not False
        or cloud.get("schema") != expected_schema
        or cloud.get("status") != "PASS"
        or cloud.get("task") != {"rotation": rotation, "fit": fit}
        or cloud.get("scientific_execution_allowed") is not True
        or cloud.get("migration_binding_sha256") != migration_sha
    ):
        raise ValueError("RGB endpoint is not bound to its assigned cloud execution")
    import torch

    saved = torch.load(Path(admission["checkpoint_path"]), map_location="cpu", weights_only=False)
    identity = saved["identity"]
    if (
        identity.get("rotation") != rotation
        or identity.get("fit_id") != fit
        or identity.get("cloud_migration_sha256") != migration_sha
        or identity.get("scientific_cloud_admission_sha256") != cloud_sha
        or identity.get("assignment_sha256") != cloud.get("assignment_sha256")
    ):
        raise ValueError("RGB checkpoint identity lacks its exact cloud ancestry")
    return {
        "producer_migration_binding_sha256": migration_sha,
        "producer_scientific_cloud_admission_sha256": cloud_sha,
        "producer_assignment_sha256": str(cloud["assignment_sha256"]),
    }


def route_endpoint(run: Path, producer_run: Path, rotation: int, fit: str) -> bool:
    """Expose one verified external endpoint under a downstream run without copying it."""
    if rotation not in {2, 3, 4} or fit not in {"R_A5", "R_C2F"}:
        raise ValueError("Only R2/R3/R4 RGB producer endpoints may be routed")
    source = producer_run / "rotations" / str(rotation) / "fits" / fit
    admission_path = source / "ENDPOINT_ADMISSION.json"
    if not admission_path.is_file():
        return False
    from operational.final_refit.scientific_fold_heads import verify_scientific_admission
    from operational.runpod_migration import runner

    admission = verify_scientific_admission(admission_path)
    if admission.get("rotation") != rotation or admission.get("fit_id") != fit:
        raise ValueError("External RGB endpoint belongs to another rotation or fit")
    cloud_identity = _verify_cloud_endpoint(producer_run, admission, rotation, fit)
    source = source.resolve(strict=True)
    destination = run / "rotations" / str(rotation) / "fits" / fit
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve(strict=True) != source:
            raise ValueError(f"RGB endpoint route is occupied by another source: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=True)
    record: dict[str, Any] = {
        "schema": "ejepa_rgb_external_endpoint_route_v1",
        "status": "PASS",
        "rotation": rotation,
        "fit": fit,
        "producer_run": str(producer_run.resolve(strict=True)),
        "source_endpoint": str(source),
        "destination_endpoint": str(destination.absolute()),
        "routing_method": "DIRECTORY_SYMLINK_NO_CHECKPOINT_COPY",
        "endpoint_admission_sha256": digest(admission_path),
        "structural_lineage_sha256": digest(source / "STRUCTURAL_LINEAGE.json"),
        "checkpoint_receipt_sha256": digest(Path(admission["checkpoint_receipt_path"])),
        "checkpoint_sha256": str(admission["checkpoint_sha256"]),
        **cloud_identity,
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "scientific_files_rewritten": False,
        "checkpoint_bytes_rewritten": False,
    }
    runner._write_once(run / f"RGB_ENDPOINT_{fit}.json", record)
    return True


__all__ = ["route_endpoint"]
