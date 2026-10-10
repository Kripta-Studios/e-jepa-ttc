"""Pin the additive lifecycle guard without rewriting historical source freezes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_io_recovery.common import validate_freeze as validate_io

ROOT = Path(__file__).resolve().parents[2]
NAME = "CONTINUITY_FREEZE.json"


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Freeze guarded producers, queue routing, and their CPU tests."""
    validate_io(run)
    quality = read_json_shared(qa)
    if quality.get("exit_code") != 0 or quality.get("junit", {}).get("failures") != "0":
        raise ValueError("Continuity admission requires passing recorded QA")
    value = {
        "schema": "rgb_port_continuity_freeze_v1",
        "event_objective_changed": False,
        "geometry_precision": "bf16_unchanged",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "files": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in [
                *sorted(Path(__file__).parent.glob("*.py")),
                ROOT / "tests/test_rgb_port_continuity.py",
                *sorted((ROOT / "operational/rgb_port_streaming").glob("*.py")),
                ROOT / "tests/test_rgb_port_streaming.py",
            ]
        },
        "upstream": {
            name: sha256_file(run / name)
            for name in (
                "IO_RECOVERY_FREEZE.json",
                "AUDIT_CODE_MIGRATION.json",
            )
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path) -> dict[str, Any]:
    """Reject any unrecorded source change; retain all upstream validators."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("Continuity source or upstream identity changed")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()
    result = payload(args.run, args.qa)
    path = args.run / NAME
    if path.exists() and read_json_shared(path) != result:
        raise ValueError("Existing continuity freeze differs")
    atomic_write_json(path, result)
