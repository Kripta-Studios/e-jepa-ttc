"""Pin the additive repair while retaining every historical freeze."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_rgb_overlap.contracts import NAME as PARENT_NAME
from operational.rgb_port_rgb_overlap.contracts import validate as validate_parent

ROOT = Path(__file__).resolve().parents[2]
NAME = "RGB_SIGNATURE_FREEZE.json"


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Require real H/V identity validation and regression evidence before routing."""
    validate_parent(run)
    quality = read_json_shared(qa)
    if quality.get("status") != "PASSED" or quality.get("optimizer_updates") != 0:
        raise ValueError("Signature repair requires passing zero-update QA")
    for key in ("all_h_v_identities_consistent", "legacy_hash_parity", "completed_fits_unchanged"):
        if quality.get(key) is not True:
            raise ValueError(f"Signature repair requires {key}")
    value = {
        "schema": "rgb_port_signature_freeze_v1",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "parent_sha256": sha256_file(run / PARENT_NAME),
        "objective_changed": False,
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                ROOT / "tests/test_rgb_port_signature_fix.py",
            ]
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path) -> dict[str, Any]:
    """Reject altered source/QA and require repair bytes to exist in committed Git."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("Signature repair admission changed")
    for relative, expected in value["files"].items():
        data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Signature repair differs from committed bytes: {relative}")
    return value
