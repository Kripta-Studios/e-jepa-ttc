"""Continue the admitted TRAIN40 queue with the parity-checked H8 execution route."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import ROOT, Lease


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Change only the H8 feature implementation; retain every original task and gate."""
    from operational.train40_system import controller_c2f_graph, controller_overlap
    from operational.train40_system.history_resources8_fast import verify_admission

    verify_admission(output)
    previous = controller_c2f_graph._launch

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
        if module == "operational.train40_system.history_features" and tag == "h8_features":
            if arguments[-2:] != ["--kind", "H8"]:
                raise ValueError("Unexpected H8 queue arguments")
            return controller_overlap._launch(
                output, tag, "operational.train40_system.history_resources8_fast", arguments
            )
        return previous(output, tag, module, arguments)

    controller_c2f_graph._launch = launch
    try:
        controller_c2f_graph.run(output, raw_root, teacher_path)
    finally:
        controller_c2f_graph._launch = previous


def main() -> None:
    """Take the unchanged controller lease before adopting or starting existing tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve() / "controller"):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())


if __name__ == "__main__":
    main()
