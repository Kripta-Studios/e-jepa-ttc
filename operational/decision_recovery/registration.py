"""Freeze sources, exposed populations, and lineage before any new scientific fit."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    RUN,
    SOURCE_ROOT,
    SOURCE_RUN,
    load_population,
    read_json,
    sha,
    source_bindings,
    write_json,
)

PROTOCOL = ROOT / "configs/decision_recovery/protocol.json"
REGISTRATION = RUN / "REGISTRATION.json"


def source_files() -> list[Path]:
    return [
        PROTOCOL,
        *[
            ROOT / "operational/decision_recovery" / name
            for name in (
                "__init__.py",
                "common.py",
                "registration.py",
                "diagnostics.py",
                "router.py",
                "campaign.py",
            )
        ],
        ROOT / "operational/range_transfer/analyze.py",
        ROOT / "operational/rgb_port/evaluate.py",
        ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]


def ancestry() -> dict[str, Any]:
    """Bind task-specific P-only origin and distinguish historical generic teachers."""
    owner = ROOT.parent / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    provenance_path = owner / "fits/E_A5_MATCHED/RUN_PROVENANCE.json"
    provenance = read_json(provenance_path)
    if (
        provenance["context"] != "P_only_fixed_epoch18"
        or provenance["source_identity"]["role"] != "P"
    ):
        raise ValueError("Historical A5 is not the declared P-only parent")
    p, h = load_population("P", ["A5"])["A5"], load_population("H", ["A5"])["A5"]
    if set(p.group_id) & set(h.group_id):
        raise ValueError("Producer training and router development groups overlap")
    evidence_paths = [
        provenance_path,
        owner / "P_MANIFEST.json",
        owner / "H_MANIFEST.json",
        SOURCE_RUN / "ADMISSION.json",
        SOURCE_RUN / "PRIMARY_ENDPOINTS.json",
        SOURCE_RUN / "adaptive/C1_geometry/REGISTRATION.json",
        SOURCE_RUN / "adaptive/B1_correspondence/REGISTRATION.json",
        SOURCE_RUN / "adaptive/A1_adaptation_depth/REGISTRATION.json",
        SOURCE_RUN / "adaptive/A1_adaptation_depth/ENDPOINTS.json",
    ]
    return {
        "task_specific_training_role": "P",
        "P_groups": sorted(p.group_id.unique().tolist()),
        "H_groups": sorted(h.group_id.unique().tolist()),
        "task_specific_groups_disjoint": True,
        "generic_teacher": provenance["ancestry"]["generic_dino"],
        "generic_teacher_limit": (
            "Historical generic teacher provenance retained; exhaustive external "
            "pretraining membership is not independently certified here."
        ),
        "H_exposure": (
            "Architecture and experiment development exposed; "
            "router group exclusion is not confirmatory validation."
        ),
        "lineage": {
            "A5": "fresh_seed7 task-specific P training with generic teacher",
            "NATIVE": "A5 -> P native adaptation",
            "CORR": "A5 -> P pair adaptation",
            "C1": "A5 -> P geometric pair adaptation",
            "B1": "C1 -> P correspondence -> P TTC",
            "FULL": "C1 -> P correspondence stage1 -> P FULL TTC",
            "LAST": "C1 -> P correspondence stage1 -> P last-block TTC",
        },
        "evidence": {str(path): sha(path) for path in evidence_paths},
    }


def register() -> dict[str, Any]:
    """Registration is immutable; an already registered run must revalidate unchanged bytes."""
    if REGISTRATION.exists():
        return verify()
    protocol = read_json(PROTOCOL)
    names = protocol["diagnostics"]["experts"]
    populations, bindings = {}, {}
    for role in ("P", "H"):
        tables = load_population(role, names)
        populations[role] = {
            "rows": len(tables["A5"]),
            "groups": sorted(tables["A5"].group_id.unique().tolist()),
        }
        bindings.update(source_bindings(role, names))
    record = {
        "schema": "v19_decision_recovery_registration_v1",
        "utc": datetime.now(UTC).isoformat(),
        "status": "REGISTERED_BEFORE_NEW_ROUTER_OR_LOSS_FITS",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protocol": protocol,
        "source_root": str(SOURCE_ROOT),
        "sources": {path.relative_to(ROOT).as_posix(): sha(path) for path in source_files()},
        "populations": populations,
        "predictions": bindings,
        "ancestry": ancestry(),
        "submission_authorized_by_this_registration": False,
        "system_promotion": False,
    }
    write_json(REGISTRATION, record)
    write_json(RUN / "EXPERT_ANCESTRY.json", record["ancestry"])
    return record


def verify() -> dict[str, Any]:
    record = read_json(REGISTRATION)
    if read_json(PROTOCOL) != record["protocol"]:
        raise ValueError("Prospective protocol changed; register a separate amendment")
    for name, expected in record["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Registered executable source changed: {name}")
    for name, expected in record["predictions"].items():
        if sha(SOURCE_ROOT / name) != expected:
            raise ValueError(f"Historical prediction evidence changed: {name}")
    for name, expected in record["ancestry"]["evidence"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Historical ancestry evidence changed: {name}")
    return record
