"""Pin an additive external mask repair without rewriting historical source freezes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_fusion_fix.contracts import NAME as PARENT_NAME
from operational.rgb_port_fusion_fix.contracts import validate as validate_parent

ROOT = Path(__file__).resolve().parents[2]
NAME = "EXTERNAL_MASK_FREEZE.json"


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Require Native producer parity on real training inputs and retained completed fits."""
    validate_parent(run)
    quality = read_json_shared(qa)
    for key in ("real_producer_parity", "completed_fits_unchanged", "tests_passed"):
        if quality.get(key) is not True:
            raise ValueError(f"External mask repair requires {key}")
    if quality.get("optimizer_updates") != 0:
        raise ValueError("Fusion QA must not consume training updates")
    value = {
        "schema": "rgb_port_external_mask_freeze_v1",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "parent_sha256": sha256_file(run / PARENT_NAME),
        "objective_changed": False,
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                ROOT / "tests/test_rgb_port_external_fix.py",
                ROOT / "operational/rgb_port_streaming/contracts.py",
            ]
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path) -> dict[str, Any]:
    """Reject changes to repair, reference implementation, tests, or recorded QA."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("External mask repair admission changed")
    for relative, expected in value["files"].items():
        data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"External mask repair differs from committed bytes: {relative}")
    return value
