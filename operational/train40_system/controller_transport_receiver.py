"""Route frozen H8 transport extraction through the admitted monitor receiver."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from operational.efficient_context.common import ROOT, Lease


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Replace only the exact frozen transport child launch and restore afterward."""

    from operational.train40_system import controller_overlap, controller_transport_graph
    from operational.train40_system.history_resources8_transport_receiver import (
        verify_admission,
    )

    verify_admission(output)
    previous = controller_overlap._launch

    def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
        if module == "operational.train40_system.history_resources8_transport_graph":
            if tag != "h8_features" or arguments[-2:] != ["--kind", "H8"]:
                raise ValueError("Unexpected H8 transport receiver arguments")
            return previous(
                output,
                tag,
                "operational.train40_system.history_resources8_transport_receiver",
                arguments,
            )
        return previous(output, tag, module, arguments)

    controller_overlap._launch = launch
    try:
        controller_transport_graph.run(output, raw_root, teacher_path)
    finally:
        controller_overlap._launch = previous


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
