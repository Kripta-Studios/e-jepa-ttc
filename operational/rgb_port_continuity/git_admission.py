"""Admit an explicit descendant Git commit while retaining scientific source freeze."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

from .contracts import NAME, ROOT, validate

ADMISSION = "CONTINUITY_GIT_ADMISSION.json"


def inspect(run: Path) -> dict[str, Any]:
    """Require exact admitted source bytes in both worktree and committed objects."""
    freeze = validate(run)
    base = read_json_shared(run / "SOURCE_FREEZE.json")["base_commit"]

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=ROOT)

    head = git("rev-parse", "HEAD").decode().strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT, check=True)
    for path, expected in freeze["files"].items():
        if (
            hashlib.sha256(git("show", head + ":" + path.replace("\\", "/"))).hexdigest()
            != expected
        ):
            raise ValueError(f"Admitted source differs from committed bytes: {path}")
    return {
        "schema": "rgb_port_continuity_git_admission_v1",
        "original_base": base,
        "admitted_head": head,
        "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "continuity_freeze_sha256": sha256_file(run / NAME),
        "original_freeze_sha256": sha256_file(run / "SOURCE_FREEZE.json"),
        "scientific_objective_changed": False,
    }


def validate_git(run: Path) -> dict[str, Any]:
    expected = read_json_shared(run / ADMISSION)
    if expected != inspect(run):
        raise ValueError("Git transition has not been admitted for these exact source bytes")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    receipt = inspect(args.run)
    if (args.run / ADMISSION).exists():
        if read_json_shared(args.run / ADMISSION) != receipt:
            raise ValueError("Existing Git admission differs; preserve it before a new transition")
    atomic_write_json(args.run / ADMISSION, receipt)
