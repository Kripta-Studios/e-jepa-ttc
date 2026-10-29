"""Compose pause-safe TRAIN40 execution with admitted relation-map reuse."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import controller_overlap, controller_pause_safe
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

_launch = controller_overlap.launch


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Route only exact original A5/C2F tasks through the reuse wrapper."""
    if module == "operational.train40_system.engine":
        if tag not in {"a5", "c2f"} or arguments != ["--arm", tag]:
            raise ValueError("Unexpected TRAIN40 engine task arguments")
        return controller_overlap._launch(
            output,
            tag,
            "operational.train40_system.engine_relation_reuse",
            arguments,
        )
    return _launch(output, tag, module, arguments)


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Install one exact engine route around the published pause-safe queue."""
    for name in (
        "COORDINATION_FREEZE.json",
        "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "IDLE_WAIT_FREEZE.json",
        "RELATION_REUSE_FREEZE.json",
    ):
        verify_sources(read(output / name))
    frozen_launch = controller_overlap.launch
    controller_overlap.launch = launch
    try:
        controller_pause_safe.run(output, raw_root, teacher_path)
    finally:
        controller_overlap.launch = frozen_launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease((args.output.resolve() / "controller").resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())
