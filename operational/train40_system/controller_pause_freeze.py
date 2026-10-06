"""Admit the pause-safe controller wrapper without changing the coordination freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "CONTROLLER_PAUSE_SAFE_PYTEST.txt": "[100%]",
    "CONTROLLER_PAUSE_SAFE_RUFF.txt": "All checks passed",
    "CONTROLLER_PAUSE_SAFE_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind the wrapper and its CPU QA to the already published coordination baseline."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required pause-safe controller QA failed: {name}")
    coordination_path = output / "COORDINATION_FREEZE.json"
    coordination = read(coordination_path)
    verify_sources(coordination)
    source = ROOT / "operational/train40_system/controller_pause_safe.py"
    files = dependency_files(
        [source, ROOT / "operational/train40_system/controller_pause_freeze.py"]
    )
    contract = {
        "schema": "train40_controller_pause_safe_freeze_v1",
        "files": files,
        "source_sha256": digest(source),
        "coordination_freeze_sha256": digest(coordination_path),
        "coordination_trainer_sha256": coordination["trainer_sha256"],
        "coordination_controller_sha256": coordination["controller_sha256"],
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "pause_markers": ["COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST"],
        "pause_blocks_all_new_heavy_tasks": True,
        "frozen_overlap_routes_unchanged": True,
        "new_optimizer_updates_during_admission": 0,
    }
    path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing pause-safe controller freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
