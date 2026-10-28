"""Exact owned-byte counting using directory metadata already supplied by Windows scandir."""

from __future__ import annotations

import os
import time
from pathlib import Path

from .common import ROOT, Campaign, digest, read


def owned_bytes(root: Path) -> int:
    """Count the same readable files as Path.rglob/is_file/stat, without symlink recursion."""
    pending = [root]
    total = 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file():
                    total += entry.stat().st_size
    return total


def install() -> None:
    """Populate the existing five-second count cache before invoking every original budget check."""
    from . import budget

    original = budget.require

    def require(c: Campaign) -> None:
        key = str(c.out)
        stamp, _ = budget._artifact_scans.get(key, (float("-inf"), 0))
        if time.monotonic() - stamp >= 5:
            used = owned_bytes(c.out)
            budget._artifact_scans[key] = time.monotonic(), used
        original(c)

    budget.require = require


def install_if_sealed() -> None:
    """Activate only the source-verified seal bound to this authorized campaign."""
    config = read(ROOT / "configs/campaign/efficient_context_v1.json")
    out = ROOT / config["new_output_root"]
    path = out / "garl/QUOTA_SCANNER_FREEZE.json"
    if not path.exists():
        return
    seal = read(path)
    if seal["parent_protocol_sha256"] != digest(out / "PROTOCOL.json"):
        raise ValueError("quota scanner campaign binding changed")
    for file in seal["files"]:
        if digest(Path(file["path"])) != file["sha256"]:
            raise ValueError("sealed quota scanner source changed")
    if seal["QA_sha256"] != digest(out / "TEST_RESULTS/fast_owned_scan/QA.json"):
        raise ValueError("sealed quota scanner QA changed")
    install()
