"""Read-only admission of final P parents and native H/V feature contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import PARENTS, ROOT, git, read, sha, utc, write


def role_audit(owner: Path) -> dict[str, Any]:
    """Audit P/H/V group assignments without opening V labels."""
    roles = {}
    for role in ("P", "H", "V"):
        path = owner / f"{role}_MANIFEST.json"
        value = read(path)
        if value.get("role") != role or value.get("status") != "COMPLETE":
            raise ValueError(f"Incomplete role manifest: {path}")
        roles[role] = {
            "path": str(path),
            "sha256": sha(path),
            "groups": value["group_ids"],
            "rows": value["population_size"],
            "split_assignment_sha256": value["split_assignment_sha256"],
        }
    for first, second in (("P", "H"), ("P", "V"), ("H", "V")):
        if set(roles[first]["groups"]) & set(roles[second]["groups"]):
            raise ValueError(f"Role group leakage: {first}/{second}")
    if len({r["split_assignment_sha256"] for r in roles.values()}) != 1:
        raise ValueError("Role split identities differ")
    return {"status": "DISJOINT", "roles": roles, "V_labels_opened": False}


def inspect(owner: Path, run: Path) -> dict[str, Any]:
    """Snapshot owner state and list exact blockers, without becoming its supervisor."""
    owner = owner.resolve(strict=True)
    repo = owner.parent.parent
    audit = role_audit(owner)
    write(run / "ROLE_AUDIT.json", audit)
    pending, parents = [], {}
    for name, endpoint in PARENTS.items():
        receipt_path = owner / "fits" / name / "CHECKPOINT_RECEIPT.json"
        receipt = read(receipt_path) if receipt_path.is_file() else {}
        freeze_path = receipt_path.parent / "ENDPOINT_FREEZE.json"
        valid = (
            receipt.get("status") == "COMPLETE"
            and receipt.get("scientific_endpoint") is True
            and receipt.get("completed_updates") == endpoint
            and freeze_path.is_file()
        )
        parents[name] = {
            "receipt_path": str(receipt_path),
            "receipt": receipt,
            "freeze_path": str(freeze_path),
            "final_admissible": valid,
        }
        if not valid:
            pending.append(
                {
                    "id": name,
                    "path": str(freeze_path),
                    "completed_updates": receipt.get("completed_updates", 0),
                    "required_updates": endpoint,
                    "owner": "V13",
                }
            )
    for name, relative in {
        "H_NATIVE_CACHE": "features/H/E_HISTORY.npz",
        "H_CACHE_RECEIPT": "features/H/E_HISTORY.json",
        "H_NORMALIZER": "features/H/E_NORMALIZER.json",
        "H_NATIVE_ADMISSION": "admissions/E_CTX_MATCHED.json",
        "NATIVE_PARENT_FREEZE": "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json",
    }.items():
        path = owner / relative
        if not path.is_file():
            pending.append({"id": name, "path": str(path), "owner": "V13"})
    observed = {
        "utc": utc(),
        "v13_head": git(repo, "rev-parse", "HEAD"),
        "v14_base": git(ROOT, "rev-parse", "HEAD"),
        "parents": parents,
        "owner": (
            read(owner / "OWNER.json")
            if (owner / "OWNER.json").is_file()
            else {"status": "NO_OWNER_FILE_OBSERVED", "ownership_acquired_by_v14": False}
        ),
        "pending": pending,
    }
    for relative in (
        "CONTINUITY_GIT_ADMISSION.json",
        "RGB_PORT_STATE.json",
        "R1_READONLY_FINAL_OBSERVATION.json",
    ):
        path = owner / relative
        if path.is_file():
            observed[relative] = {"path": str(path), "sha256": sha(path), "value": read(path)}
    write(run / "OWNER_OBSERVATION.json", observed)
    write(run / "PENDING_DEPENDENCIES.json", {"utc": utc(), "dependencies": pending})
    return observed


def admit(owner: Path, run: Path) -> dict[str, Any] | None:
    """Admit only final endpoints bound to P and a native sealed H admission."""
    observation = inspect(owner, run)
    if observation["pending"]:
        return None
    files: dict[str, str] = {}
    parents = {}
    for name in PARENTS:
        folder = owner / "fits" / name
        receipt = read(folder / "CHECKPOINT_RECEIPT.json")
        freeze = read(folder / "ENDPOINT_FREEZE.json")
        if freeze.get("fit_id") != name or receipt.get("fit_id") != name:
            raise ValueError(f"Parent identity mismatch: {name}")
        for basename, expected in freeze["files"].items():
            path = folder / basename
            if path.parent != folder or sha(path) != expected:
                raise ValueError(f"Frozen parent file changed: {path}")
            files[str(path.resolve())] = expected
        checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
        if checkpoint != (folder / "checkpoint_last.pt").resolve():
            raise ValueError("Parent checkpoint is outside its owner directory")
        if sha(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint mismatch")
        provenance = read(folder / "RUN_PROVENANCE.json")
        if provenance["dataset_manifest_hash"] != sha(owner / "P_MANIFEST.json"):
            raise ValueError("Parent was not trained under the frozen P-only role")
        parents[name] = receipt["checkpoint_sha256"]
    native_path = owner / "admissions/E_CTX_MATCHED.json"
    native = read(native_path)
    cache = owner / "features/H/E_HISTORY.npz"
    normalizer = owner / "features/H/E_NORMALIZER.json"
    checks = {
        "source_sha256": cache,
        "normalizer_sha256": normalizer,
        "role_manifest_sha256": owner / "H_MANIFEST.json",
        "split_sha256": owner / "SPLIT_MANIFEST.json",
    }
    for key, path in checks.items():
        if native.get(key) != sha(path):
            raise ValueError(f"Native H admission mismatch: {key}")
        files[str(path.resolve())] = sha(path)
    norm = read(normalizer)
    from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256

    if (
        norm.get("fit_role") != "H"
        or norm.get("modality") != "event"
        or norm.get("schema_sha256") != EVENT_PHASE17_SHA256
        or native.get("precision") != "float32"
    ):
        raise ValueError("Native normalizer/schema/precision mismatch")
    # The native all-modality freeze is read if present; no RGB training is requested.
    parent_path = owner / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    if sha(parent_path) != native["parent_sha256"] or sha(parent_path) != norm["producer_sha256"]:
        raise ValueError("Native H parent ancestry differs")
    for name, expected in parents.items():
        if read(parent_path)["endpoints"][name]["checkpoint_sha256"] != expected:
            raise ValueError("H cache was built from different event parents")
    history = read(cache.with_suffix(".json"))
    if (
        history.get("status") != "COMPLETE"
        or history.get("role") != "H"
        or history.get("modality") != "event"
        or history.get("output_sha256") != sha(cache)
        or history.get("normalizer_sha256") != sha(normalizer)
    ):
        raise ValueError("Native history receipt mismatch")
    for path in (native_path, parent_path, cache.with_suffix(".json")):
        files[str(path.resolve())] = sha(path)
    return {
        "status": "ADMITTED",
        "owner_run": str(owner.resolve()),
        "parents": parents,
        "cache_path": str(cache.resolve()),
        "normalizer_path": str(normalizer.resolve()),
        "files": files,
        "native_device": native["device"],
        "new_device": "cpu",
        "precision": "float32",
        "native_endpoint_reused": False,
    }


def revalidate(binding: dict[str, Any]) -> None:
    """Recheck immutable admitted files at resume/endpoint boundaries."""
    for path, expected in binding["files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Admitted source changed: {path}")
