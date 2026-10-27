"""Admission rules for additive local-cache scientific overlay v6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operational.runpod_migration.inventory import digest
from scripts.runpod_scientific_overlay_v6 import _validate_proofs


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_v6_requires_matching_zero_update_input_and_eight_update_resume(tmp_path: Path) -> None:
    readiness = _write(tmp_path / "READY.json", {"status": "READY"})
    ready_sha = digest(readiness)
    route = {
        "readiness_sha256": ready_sha,
        "content_identity_sha256": "a" * 64,
        "route_map_sha256": "b" * 64,
    }
    input_path = _write(
        tmp_path / "INPUT.json",
        {
            "schema": "ejepa_runpod_local_cache_preflight_v1",
            "status": "PASS",
            "rotation": 3,
            "fit": "R_A5",
            "optimizer_updates": 0,
            "training_called": False,
            "scientific_execution_allowed": False,
            "logical_vs_local_bit_exact": True,
            "python_numpy_torch_rng_unchanged": True,
            "routing": route,
        },
    )
    resume = {
        "schema": "ejepa_local_cache_resume_probe_v1",
        "status": "PASS",
        "rotation": 3,
        "fit": "R_A5",
        "technical_optimizer_updates": 8,
        "scientific_optimizer_updates": 0,
        "scientific_execution_allowed": False,
        "readiness_sha256": ready_sha,
        "input_proof": {
            "logical_vs_local_bit_exact": True,
            "python_numpy_torch_rng_unchanged": True,
            "routing": route,
        },
    }
    resume_path = _write(tmp_path / "RESUME.json", resume)
    _validate_proofs(input_path, resume_path, readiness)
    resume["technical_optimizer_updates"] = 7
    _write(resume_path, resume)
    with pytest.raises(ValueError, match="incomplete"):
        _validate_proofs(input_path, resume_path, readiness)
