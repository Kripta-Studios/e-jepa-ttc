"""Candidate byte identity is distinct from scientific ancestry and exposure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import canonical_sha, file_sha, read_document

MODEL_IDS = (
    "E_A5",
    "E_C2F",
    "R_A5",
    "R_C2F",
    "PAIR_E",
    "PAIR_R",
    "E_CTX_MATCHED",
    "R_CTX",
    "PHASE_ONLY_EVENT_S7_TRAIN40",
)
SCOPES = (
    "training_groups",
    "checkpoint_selection_groups",
    "architecture_selection_groups",
    "normalization_fit_groups",
)


def evaluate_ancestry(component: dict[str, Any], exclusions: set[str]) -> dict[str, Any]:
    """Never interpret an empty or missing scope as verified absence of exposure."""
    exposure = sorted(
        {group for scope in SCOPES for group in (component.get(scope) or []) if group in exclusions}
    )
    unresolved = [scope for scope in SCOPES if not isinstance(component.get(scope), list)]
    if component.get("ancestry_complete") is not True or not component.get("evidence_refs"):
        unresolved.append("ancestry_not_evidenced")
    parents = component.get("parent_artifact_hashes")
    if parents is None:
        unresolved.append("unknown_parents")
    elif parents:
        # A declarative inventory alone cannot certify that recursive parent audits ran.
        unresolved.append("recursive_parent_audit_required")
    if "teacher_artifact_hash" not in component or component.get("teacher_artifact_hash"):
        unresolved.append("teacher_ancestry")
    return dict(
        component,
        exposure_status="exposed" if exposure else "unresolved" if unresolved else "clear",
        excluded_groups_seen=exposure,
        unresolved=unresolved,
    )


def audit_lineage(
    candidate: Path, inventory: dict[str, Any], exclusions: dict[str, Any]
) -> dict[str, Any]:
    """Verify portable files and enumerate all nine components, preserving unknown scopes."""
    paths = read_document(candidate / "PORTABLE_PATHS.json")
    if paths.get("schema") != "ejepa_final_candidate_local_paths_v1":
        raise ValueError("Unknown portable candidate schema")
    root = candidate.resolve(strict=True)
    binding = {}
    for name in (
        *MODEL_IDS,
        "event_normalizer",
        "rgb_normalizer",
        "MODEL_SEAL.json",
        "OOF_RECIPE.json",
    ):
        entry = paths["components"][name]
        path = Path(entry["local_path"]).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha(path) != entry["sha256"]:
            raise ValueError(f"Candidate identity mismatch: {name}")
        binding[name] = file_sha(path)
    seal = read_document(candidate / "MODEL_SEAL.json")
    if seal["seal_identity_sha256"] != canonical_sha(
        {k: v for k, v in seal.items() if k != "seal_identity_sha256"}
    ):
        raise ValueError("Invalid model seal identity")
    expected = dict(seal["deployment_dependencies"])
    expected[MODEL_IDS[-1]] = seal["trained_endpoints"][MODEL_IDS[-1]]["checkpoint_sha256"]
    if any(binding[name] != expected[name] for name in MODEL_IDS):
        raise ValueError("Candidate differs from sealed dependency graph")
    for modality in ("event", "rgb"):
        if binding[modality + "_normalizer"] != seal["normalizers"][modality]:
            raise ValueError("Normalizer differs from seal")
    records = inventory["components"]
    ids = [r["component_id"] for r in records]
    if len(ids) != 9 or set(ids) != set(MODEL_IDS):
        raise ValueError("Inventory must contain the nine unique deployed components")
    excluded = set(exclusions["excluded_recordings"])
    rows = []
    for component in records:
        name = component["component_id"]
        if component.get("artifact_sha256") != binding[name]:
            raise ValueError(f"Inventory hash differs: {name}")
        rows.append(evaluate_ancestry(component, excluded))
    return {
        "schema": "evttc_transfer_lineage_v1",
        "identity_status": "verified",
        "status": "verified"
        if all(r["exposure_status"] == "clear" for r in rows)
        else "unresolved",
        "binding": binding,
        "components": rows,
        "inventory_sha256": canonical_sha(inventory),
        "exclusions_sha256": canonical_sha(exclusions),
        "limitations": [
            "Hashes establish identity, not clean ancestry; frozen does not mean unexposed.",
            "Parent and teacher declarations require independent recursive evidence.",
        ],
    }
