"""Admit an additive RGB input runtime without rewriting scientific freezes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_loader_v3.contracts import NAME as PARENT_NAME
from operational.rgb_port_loader_v3.contracts import validate as validate_parent

ROOT = Path(__file__).resolve().parents[2]
NAME = "RGB_LOADER_FREEZE.json"
MAX_DINO_SHARDS = 32
MAX_DINO_BYTES = 64 * 1024**2


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Bind exact parity evidence, bounded cache and the retained parent runtime."""
    validate_parent(run)
    quality = read_json_shared(qa)
    if quality.get("status") != "PASSED" or quality.get("exact_batch_parity") is not True:
        raise ValueError("RGB loader admission requires passing QA and exact real batch parity")
    value = {
        "schema": "rgb_port_rgb_loader_freeze_v1",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "parent_sha256": sha256_file(run / PARENT_NAME),
        "fit_ids": ["R_A5", "R_C2F"],
        "max_dino_shards": MAX_DINO_SHARDS,
        "max_dino_bytes": MAX_DINO_BYTES,
        "prefetch_depth": 0,
        "collate": "canonical_torch_stack_six_consumed_fields",
        "geometry_precision": "bf16_unchanged",
        "objective_changed": False,
        "commit_reserve_gib": 1,
        "commit_reserve_authorization": "explicit_user_request_20261010",
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                *sorted((ROOT / "tests").glob("test_rgb_port_rgb_loader*.py")),
            ]
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path, *, committed: bool = True) -> dict[str, Any]:
    """Reject unadmitted source changes and preserve every parent validator."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("RGB loader implementation or upstream freeze changed")
    if committed:
        for relative, expected in value["files"].items():
            data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"RGB loader differs from committed bytes: {relative}")
    return value
