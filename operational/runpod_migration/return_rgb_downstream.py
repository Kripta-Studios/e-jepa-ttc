"""Validate archived RGB head authority without rewriting original provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .inventory import digest, read


def verify_downstream_authority(root: Path, repository: Path) -> dict[str, Any]:
    """Verify the byte-pinned archive closure and locally authorized RGB task."""
    root = root.resolve(strict=True)
    repository = repository.resolve(strict=True)
    if not root.is_relative_to(repository / "artifacts/runpod_returns"):
        raise ValueError("RGB returns must remain isolated from the active run")
    archive = root.parents[1]
    manifest = read(archive / "MANIFEST.json")
    if (
        manifest.get("schema") != "ejepa_cloud_rgb_complete_archive_v1"
        or manifest.get("status") != "READY"
    ):
        raise ValueError("RGB heads require the complete immutable return archive")
    prefix = root.relative_to(archive).as_posix() + "/"
    expected = {
        path[len(prefix) :]: item
        for path, item in manifest["files"].items()
        if path.startswith(prefix)
    }
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if not expected or set(expected) != actual:
        raise ValueError("RGB downstream return file coverage differs")
    for relative, item in expected.items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(root)
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError(f"RGB downstream archive member changed: {relative}")
    admission = read(root / "RGB_DOWNSTREAM_ADMISSION.json")
    task = admission.get("task", {})
    rotation = task.get("rotation")
    if rotation not in {2, 3, 4} or task.get("fit") != "RGB_DOWNSTREAM":
        raise ValueError("Unknown RGB downstream assignment")
    authority = repository / f"artifacts/runpod_migration_20261026/r{rotation}_rgb_downstream"
    assignment_path = authority / f"SCIENTIFIC_R{rotation}_RGB_DOWNSTREAM.json"
    assignment = read(assignment_path)
    migration_sha = digest(root / "MIGRATION_BINDING.json")
    migration = read(root / "MIGRATION_BINDING.json")
    scientific_sha = digest(
        repository / "artifacts/final_refit_20261019_scientific_oof/SCIENTIFIC_CONFIG.json"
    )
    bindings = {
        "assignment_sha256": assignment_path,
        "budget_sha256": authority / "USER_BUDGET_24USD.json",
        "exclusion_sha256": authority / "LOCAL_QUEUE_EXCLUSION.json",
        "migration_binding_sha256": root / "MIGRATION_BINDING.json",
        "external_deployment_binding_sha256": root / "RGB_DEPLOYMENT_BINDING.json",
        "git_attribution_sha256": root / "RGB_HEAD_GIT_ATTRIBUTION.json",
        "queue_entrypoint_sha256": repository / "scripts/runpod_rgb_queue_v1.py",
        "queue_library_sha256": repository / "operational/runpod_migration/rgb_queue.py",
        "queue_endpoint_router_sha256": (
            repository / "operational/runpod_migration/rgb_queue_endpoints.py"
        ),
    }
    if (
        admission.get("schema") != "ejepa_cloud_rgb_downstream_admission_v1"
        or admission.get("status") != "PASS"
        or admission.get("scientific_execution_allowed") is not True
        or assignment.get("tasks") != [task]
        or assignment.get("execution_authorized") is not True
        or assignment.get("migration_binding_sha256") != migration_sha
        or migration.get("schema") != "ejepa_linux_technical_migration_binding_v3"
        or migration.get("seed") != 7
        or migration.get("precision") != "bf16"
        or migration.get("checkpoint_updates") != 100
        or migration.get("scientific_files_rewritten") is not False
        or migration.get("original_authority_sha256") != scientific_sha
        or digest(root / "SCIENTIFIC_CONFIG.json") != scientific_sha
        or any(admission.get(key) != digest(path) for key, path in bindings.items())
    ):
        raise ValueError("RGB heads differ from locally authorized scientific execution")
    deployment = read(root / "RGB_DEPLOYMENT_BINDING.json")
    source_locations = {
        "runpod_rgb_bootstrap_v1.py": repository / "scripts/runpod_rgb_bootstrap_v1.py",
        "runpod_rgb_queue_v1.py": repository / "scripts/runpod_rgb_queue_v1.py",
        **{
            f"{name}.py": repository / f"operational/runpod_migration/{name}.py"
            for name in ("rgb_queue", "rgb_queue_endpoints", "rgb_downstream_coverage")
        },
    }
    source_hashes = deployment.get("source_sha256", {})
    if (
        deployment.get("schema") != "ejepa_rgb_external_deployment_v1"
        or deployment.get("status") != "PASS"
        or deployment.get("migration_binding_sha256") != migration_sha
        or deployment.get("frozen_migration_sources_unchanged") is not True
        or {Path(name).name for name in source_hashes} != set(source_locations)
        or any(
            sha != digest(source_locations[Path(name).name]) for name, sha in source_hashes.items()
        )
    ):
        raise ValueError("RGB external deployment differs from locally reviewed code")
    result = read(root / "RGB_DOWNSTREAM_RESULT.json")
    if (
        result.get("status") != "COMPLETE"
        or result.get("stage") != "RGB_OOF"
        or result.get("rotation") != rotation
        or result.get("emitted_interface") != "PHASE17_RAW_RGB"
        or result.get("phase20_context_emitted") is not False
        or result.get("final_ttc_predictions_emitted") is not False
        or result.get("authority", {}).get("rgb_downstream_admission_sha256")
        != digest(root / "RGB_DOWNSTREAM_ADMISSION.json")
    ):
        raise ValueError("RGB raw OOF closure is incomplete or misrepresented")
    for artifact in result["artifacts"]:
        relative = artifact.get("path", artifact.get("relative_path"))
        if relative not in expected:
            raise ValueError("RGB result lists an artifact missing from the pinned archive")
        if any(artifact[key] != expected[relative][key] for key in ("sha256", "bytes")):
            raise ValueError("RGB result artifact identity differs from archived bytes")
    return {
        "status": "PASS",
        "rotation": rotation,
        "fits": ["PAIR_R", "R_CTX"],
        "migration_binding_sha256": migration_sha,
        "downstream_admission_sha256": digest(root / "RGB_DOWNSTREAM_ADMISSION.json"),
        "original_files_rewritten": False,
    }
