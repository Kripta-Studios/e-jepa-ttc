"""Resume the frozen controller with a safe metadata-only raw preparation launcher."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import controller
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

_launch = controller.launch


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Only substitute the safe disk/progress wrapper; every other launch remains identical."""
    if module == "operational.train40_system.prepare_partition":
        module = "operational.train40_system.prepare_safe"
    return _launch(output, tag, module, arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    verify_sources(read(args.output / "SAFE_PREPARATION_FREEZE.json"))
    controller.launch = launch
    with Lease((args.output / "controller").resolve()):
        controller.run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())
