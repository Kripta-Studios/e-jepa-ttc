"""Verify the single final adapted candidate before sensor inference."""

from __future__ import annotations

from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_startup.runtime import RouterEnsemble, load_router


def load_submission_model(root: Path) -> RouterEnsemble:
    """Bind upstream, fit identity, export and startup implementation bytes."""
    model = read(root / "MODEL.json")
    if model.get("schema") != "phase_evttc_submission_model_v1":
        raise ValueError("Unsupported submission model schema")
    for name, key in (
        ("upstream/BUNDLE.json", "upstream_sha256"),
        ("router/FINAL_FREEZE.json", "router_freeze_sha256"),
        ("router/router.ts", "export_sha256"),
    ):
        if file_sha(root / name) != model[key]:
            raise ValueError("Changed submission model artifact")
    expected = {
        f"operational/phase_startup/{name}.py"
        for name in ("__init__", "geometry", "model", "runtime")
    }
    normalized = {key.replace("\\", "/"): digest for key, digest in model["implementation"].items()}
    if set(normalized) != expected:
        raise ValueError("Changed implementation inventory")
    for name, digest in normalized.items():
        if file_sha(Path(name)) != digest:
            raise ValueError("Changed startup implementation")
    verified_upstream_paths(root / "upstream")
    return load_router(root / "router")
