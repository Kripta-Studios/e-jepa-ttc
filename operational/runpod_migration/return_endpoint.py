"""Verify isolated cloud snapshots and normalize paths without rewriting provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from .inventory import digest, read


class PathTranslation(Protocol):
    """Translate declared paths without changing any scientific identity field."""

    def translate(self, value: str | os.PathLike[str]) -> str: ...


def verify_snapshot_files(root: Path) -> dict[str, Any]:
    """Verify complete snapshot coverage, file identity and safe path containment."""
    root = root.resolve(strict=True)
    manifest = read(root / "MANIFEST.json")
    if manifest.get("schema") != "ejepa_cloud_checkpoint_snapshot_v1":
        raise ValueError("Unknown cloud snapshot schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Snapshot has no pinned files")
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != set(expected) | {"MANIFEST.json"}:
        raise ValueError("Snapshot file coverage differs")
    for relative, identity in expected.items():
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Snapshot member escaped its isolated root")
        if digest(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"Snapshot member changed: {relative}")
    return manifest


def normalized_role(value: object, mapper: PathTranslation) -> object:
    """Translate only the two path fields; preserve row identities and group data."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for field in ("manifest_path", "rows_path"):
        if field in result:
            if not isinstance(result[field], str):
                raise ValueError(f"Role field {field} must be a path string")
            result[field] = str(Path(mapper.translate(result[field])).resolve(strict=True))
    return result


def verify_cloud_snapshot(
    root: Path, *, repository: Path, require_endpoint: bool
) -> dict[str, Any]:
    """Cross-check returned bytes with locally retained authorizations and code."""
    root, repository = root.resolve(strict=True), repository.resolve(strict=True)
    allowed = repository / "artifacts/runpod_returns"
    if not root.is_relative_to(allowed):
        raise ValueError("Cloud data must remain in the isolated returns tree")
    manifest = verify_snapshot_files(root)
    rotation, fit = manifest.get("rotation"), manifest.get("fit")
    if rotation not in {2, 3, 4} or fit not in {"R_A5", "R_C2F"}:
        raise ValueError("Unassigned cloud task")
    task = {"rotation": rotation, "fit": fit}
    operation = repository / "artifacts/runpod_migration_20261026"
    local_authorization = operation
    if fit == "R_C2F":
        local_authorization /= f"r{rotation}_c2f"
        assignment_name = f"SCIENTIFIC_R{rotation}_C2F.json"
        entrypoint = "runpod_scientific_overlay_c2f_v1.py"
        schema = "ejepa_scientific_cloud_admission_c2f_v1"
    else:
        if rotation != 4:
            local_authorization /= f"r{rotation}"
        assignment_name = f"SCIENTIFIC_R{rotation}.json"
        entrypoint = f"runpod_scientific_overlay_v{4 if rotation == 4 else 5}.py"
        schema = f"ejepa_scientific_cloud_admission_v{4 if rotation == 4 else 5}"
    authorization = root / "cloud_authorization"
    assignment_path = authorization / assignment_name
    assignment = read(assignment_path)
    # A cloud machine cannot authorize a new task by rewriting its own receipts.
    if digest(assignment_path) != digest(local_authorization / assignment_name):
        raise ValueError("Cloud assignment differs from its local authorization")
    cloud = read(root / "SCIENTIFIC_CLOUD_ADMISSION.json")
    migration = read(root / "MIGRATION_BINDING.json")
    original_run = repository / "artifacts/final_refit_20261019_scientific_oof"
    authority_hash = digest(original_run / "SCIENTIFIC_CONFIG.json")
    if (
        migration.get("schema") != "ejepa_linux_technical_migration_binding_v3"
        or migration.get("original_authority_sha256") != authority_hash
        or digest(root / "SCIENTIFIC_CONFIG.json") != authority_hash
        or migration.get("scientific_files_rewritten") is not False
        or migration.get("seed") != 7
        or migration.get("precision") != "bf16"
        or migration.get("checkpoint_updates") != 100
        or cloud.get("schema") != schema
        or cloud.get("status") != "PASS"
        or cloud.get("task") != task
        or cloud.get("scientific_execution_allowed") is not True
        or assignment.get("tasks") != [task]
        or assignment.get("execution_authorized") is not True
        or assignment.get("migration_binding_sha256") != digest(root / "MIGRATION_BINDING.json")
    ):
        raise ValueError("Cloud task, migration or scientific authority differs")
    bindings = {
        "migration_binding_sha256": root / "MIGRATION_BINDING.json",
        "scientific_entrypoint_sha256": authorization / entrypoint,
        "assignment_sha256": assignment_path,
        "budget_sha256": authorization / "USER_BUDGET_24USD.json",
        "local_queue_exclusion_sha256": authorization / "LOCAL_QUEUE_EXCLUSION.json",
        "resume_proof_sha256": root / f"technical_resume_probe/{fit}/RESULT.json",
        "windows_compatibility_sha256": authorization / "WINDOWS_CHECKPOINT_COMPATIBILITY.json",
    }
    if any(cloud.get(key) != digest(path) for key, path in bindings.items()):
        raise ValueError("Cloud admission evidence changed")
    if cloud["scientific_entrypoint_sha256"] != digest(repository / "scripts" / entrypoint):
        raise ValueError("Cloud entrypoint differs from the locally reviewed source")
    if digest(authorization / "LOCAL_QUEUE_EXCLUSION.json") != digest(
        local_authorization / "LOCAL_QUEUE_EXCLUSION.json"
    ):
        raise ValueError("Returned ownership evidence differs from its local original")
    proof = read(bindings["resume_proof_sha256"])
    if (
        proof.get("status") != "PASS"
        or proof.get("rotation") != rotation
        or proof.get("fit") != fit
        or proof.get("technical_optimizer_updates") != 8
        or proof.get("scientific_optimizer_updates") != 0
    ):
        raise ValueError("Cloud resume proof differs from this task")
    directory = root / f"rotations/{rotation}/fits/{fit}"
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    pointer = read(directory / "CHECKPOINT_POINTER.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    checkpoint = directory / "checkpoint_last.pt"
    if not (
        receipt["completed_updates"]
        == pointer["completed_updates"]
        == journal["durable_updates"]
        == manifest["completed_updates"]
        and receipt["checkpoint_sha256"]
        == pointer["checkpoint_sha256"]
        == manifest["checkpoint_sha256"]
        == digest(checkpoint)
        and receipt["identity_sha256"] == pointer["identity_sha256"] == journal["identity_sha256"]
    ):
        raise ValueError("Returned durable checkpoint transaction differs")
    if require_endpoint and not (
        receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
        and journal["completed_updates"] == journal["durable_updates"]
        and journal["pending_update_upper"] == 0
        and (directory / "ENDPOINT_ADMISSION.json").is_file()
    ):
        raise ValueError("A partial checkpoint cannot be admitted as a scientific endpoint")
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = saved["identity"]
    from operational.final_refit.recipe import canonical_sha256

    if not (
        identity.get("technical") is False
        and identity.get("rotation") == rotation
        and identity.get("fit_id") == fit
        and identity.get("identity_sha256")
        == receipt["identity_sha256"]
        == saved["identity_sha256"]
        and canonical_sha256({k: v for k, v in identity.items() if k != "identity_sha256"})
        == identity["identity_sha256"]
        and identity.get("cloud_migration_sha256") == cloud["migration_binding_sha256"]
        and identity.get("scientific_cloud_admission_sha256")
        == digest(root / "SCIENTIFIC_CLOUD_ADMISSION.json")
        and identity.get("assignment_sha256") == cloud["assignment_sha256"]
        and identity.get("linux_resume_proof_sha256") == cloud["resume_proof_sha256"]
    ):
        raise ValueError("Checkpoint identity differs from cloud authorization")
    return {
        "status": "PASS",
        "rotation": rotation,
        "fit": fit,
        "updates": saved["completed_updates"],
        "snapshot_manifest_sha256": digest(root / "MANIFEST.json"),
        "checkpoint_sha256": digest(checkpoint),
        "endpoint_required": require_endpoint,
        "original_files_rewritten": False,
        "active_queue_modified": False,
    }


def canonical_role_equal(left: object, right: object, mapper: PathTranslation) -> bool:
    """Compare role identities after path translation only, never relaxing hashes."""
    return json.dumps(normalized_role(left, mapper), sort_keys=True) == json.dumps(
        normalized_role(right, mapper), sort_keys=True
    )
