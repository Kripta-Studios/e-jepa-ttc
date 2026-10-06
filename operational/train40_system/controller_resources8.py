"""Continue the queue with eight bounded H8 raw workers and the original live trainers."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import controller, controller_resume
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

_active = controller.active


def active(marker: str) -> bool:
    """Recognize both feature entrypoints so the original one-heavy-writer gate still holds."""
    if marker == "operational.train40_system.history_features":
        return _active(marker) or _active("operational.train40_system.history_resources8")
    return _active(marker)


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Select only the admitted H8 resource variant; retain every other task and argument."""
    if module == "operational.train40_system.history_features" and tag == "h8_features":
        if arguments[-2:] != ["--kind", "H8"]:
            raise ValueError("Unexpected H8 task arguments")
        module = "operational.train40_system.history_resources8"
    return controller_resume.launch(output, tag, module, arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    verify_sources(read(args.output / "SAFE_PREPARATION_FREEZE.json"))
    verify_sources(read(args.output / "FEATURES_EIGHT_WORKER_FREEZE.json"))
    controller.active = active
    controller.launch = launch
    with Lease((args.output / "controller").resolve()):
        controller.run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())
