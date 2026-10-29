"""Compose admitted overlap training with sealed input and bounded resource routes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import controller, controller_sealed
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

_active = controller.active
_launch = controller.launch


def active(marker: str) -> bool:
    """Recognize every admitted variant behind an original controller task marker."""
    if marker == "operational.train40_system.engine":
        return _active(marker) or _active("operational.train40_system.engine_overlap")
    if marker == "operational.train40_system.history_features":
        return _active(marker) or _active("operational.train40_system.history_resources8")
    return _active(marker)


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Route only the three admitted variants and preserve all task arguments."""
    if module == "operational.train40_system.prepare_partition":
        module = "operational.train40_system.prepare_safe"
    elif module == "operational.train40_system.history_features" and tag == "h8_features":
        if arguments[-2:] != ["--kind", "H8"]:
            raise ValueError("Unexpected H8 task arguments")
        module = "operational.train40_system.history_resources8"
    elif module == "operational.train40_system.engine":
        if tag not in {"a5", "c2f"} or arguments != ["--arm", tag]:
            raise ValueError("Unexpected TRAIN40 engine task arguments")
        module = "operational.train40_system.engine_overlap"
    return _launch(output, tag, module, arguments)


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Verify every composed admission before entering the unchanged queue."""
    for name in (
        "SAFE_PREPARATION_FREEZE.json",
        "FEATURES_EIGHT_WORKER_FREEZE.json",
        "CONTROLLER_SEALED_FREEZE.json",
        "COORDINATION_FREEZE.json",
    ):
        verify_sources(read(output / name))
    controller.refresh_complete_inputs = controller_sealed.refresh_complete_inputs
    controller.active = active
    controller.launch = launch
    controller.run(output, raw_root, teacher_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease((args.output.resolve() / "controller").resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())
