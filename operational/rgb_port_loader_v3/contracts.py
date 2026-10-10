"""Additive execution admission preserving every historical scientific freeze."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_continuity.contracts import validate as validate_continuity

ROOT = Path(__file__).resolve().parents[2]
NAME = "LOADER_V3_FREEZE.json"


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Pin implementation, passing QA, bounds and the retained parent freeze."""
    validate_continuity(run)
    quality = read_json_shared(qa)
    if quality.get("status") != "PASSED" or quality.get("exact_batch_parity") is not True:
        raise ValueError("Loader admission requires passing QA and exact real batch parity")
    value = {
        "schema": "rgb_port_loader_v3_freeze_v1",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "parent_sha256": sha256_file(run / "CONTINUITY_FREEZE.json"),
        "workers_per_fit": 2,
        "queued_batches_per_fit": 2,
        "worker_kind": "threads_sharing_numpy_memory",
        "geometry_precision": "bf16_unchanged",
        "objective_changed": False,
        "canonical_validation_unchanged": True,
        "commit_reserve_gib": 3,
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                ROOT / "tests/test_rgb_port_loader_v3.py",
            ]
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path, *, committed: bool = True) -> dict[str, Any]:
    """Require exact admitted worktree and Git bytes; reject unrecorded edits."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("Loader implementation or upstream freeze changed")
    if committed:
        for relative, expected in value["files"].items():
            data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Loader differs from committed bytes: {relative}")
    return value
