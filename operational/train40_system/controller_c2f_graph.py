"""Continue the unchanged TRAIN40 queue with the admitted C2F graph trainer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import Lease, digest
from operational.train40_system import controller_overlap, controller_pause_safe
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.engine_host_fast_4 import verify_admission

_launch = controller_overlap.launch


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Route the existing producer task without changing scientific arguments."""
    if module == "operational.train40_system.engine":
        if tag not in {"a5", "c2f"} or arguments != ["--arm", tag]:
            raise ValueError("Unexpected TRAIN40 producer arguments")
        implementation = (
            "operational.train40_system.engine_c2f_graph_replay"
            if tag == "c2f"
            else "operational.train40_system.engine_host_fast_4"
        )
        return controller_overlap._launch(output, tag, implementation, arguments)
    return _launch(output, tag, module, arguments)


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Verify graph admission and retain the existing downstream orchestration."""
    verify_admission(output)
    freeze = read(output / "C2F_GRAPH_REPLAY_FREEZE.json")
    verify_sources(freeze)
    if freeze["controller_sha256"] != digest(Path(__file__)):
        raise ValueError("C2F graph controller differs from its admitted source")
    previous = controller_overlap.launch
    controller_overlap.launch = launch
    try:
        controller_pause_safe.run(output, raw_root, teacher_path)
    finally:
        controller_overlap.launch = previous


def main() -> None:
    """Own the controller lease while the original queue launches bounded tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve() / "controller"):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())


if __name__ == "__main__":
    main()
