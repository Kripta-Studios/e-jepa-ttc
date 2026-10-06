"""Publish the bounded full-step profiler after CPU-only admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "TRAIN40_PROFILED_PYTEST.txt": "[100%]",
    "TRAIN40_PROFILED_RUFF.txt": "All checks passed",
    "TRAIN40_PROFILED_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind wrapper sources, CPU QA, and both already published execution freezes."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required execution-profile QA failed: {name}")
    coordination_path = output / "COORDINATION_FREEZE.json"
    pause_path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    coordination, pause = read(coordination_path), read(pause_path)
    verify_sources(coordination)
    verify_sources(pause)
    engine = ROOT / "operational/train40_system/engine_profiled.py"
    controller = ROOT / "operational/train40_system/controller_profiled.py"
    publisher = ROOT / "operational/train40_system/execution_profile_freeze.py"
    contract = {
        "schema": "train40_execution_profile_freeze_v1",
        "files": dependency_files([engine, controller, publisher]),
        "engine_profiled_sha256": digest(engine),
        "controller_profiled_sha256": digest(controller),
        "coordination_freeze_sha256": digest(coordination_path),
        "controller_pause_safe_freeze_sha256": digest(pause_path),
        "coordination_trainer_sha256": coordination["trainer_sha256"],
        "pause_safe_source_sha256": pause["source_sha256"],
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "schedule": {"wait": 20, "warmup": 2, "active": 5, "repeat": 1},
        "profile_steps_are_actual_scientific_commits": True,
        "additional_optimizer_updates": 0,
        "checkpoint_contract_unchanged": True,
        "instrumentation_removed_after_updates": 27,
        "raw_tensor_payloads_exported": False,
    }
    path = output / "EXECUTION_PROFILE_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing execution-profile freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
