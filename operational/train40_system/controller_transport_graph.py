"""Continue the frozen queue with the admitted H8 transport graph entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import ROOT, Lease


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Route only the original H8 feature task through transport replay."""

    from operational.train40_system import controller_c2f_graph, controller_overlap
    from operational.train40_system.history_resources8_transport_graph import verify_admission

    verify_admission(output)
    previous = controller_c2f_graph._launch

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
        if module == "operational.train40_system.history_features" and tag == "h8_features":
            if arguments[-2:] != ["--kind", "H8"]:
                raise ValueError("Unexpected H8 queue arguments")
            return controller_overlap._launch(
                output,
                tag,
                "operational.train40_system.history_resources8_transport_graph",
                arguments,
            )
        return previous(output, tag, module, arguments)

    controller_c2f_graph._launch = launch
    try:
        controller_c2f_graph.run(output, raw_root, teacher_path)
    finally:
        controller_c2f_graph._launch = previous


def main() -> None:
    """Own the existing controller lease before adopting the unchanged queue."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve() / "controller"):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())


if __name__ == "__main__":
    main()
