"""Keep the frozen overlap queue paused while a user coordination marker exists."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import controller_overlap
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

ENGINE_MARKER = "operational.train40_system.engine"
PAUSE_MARKERS = ("COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST")


def pause_requested(output: Path) -> bool:
    """Observe user-owned markers without reading, changing or removing their contents."""
    return any((output / name).is_file() for name in PAUSE_MARKERS)


def active_for(output: Path, delegate: Callable[[str], bool]) -> Callable[[str], bool]:
    """Treat a persistent pause as an active engine so the one-heavy gate stays closed."""

    def active(marker: str) -> bool:
        if marker == ENGINE_MARKER and pause_requested(output):
            return True
        return delegate(marker)

    return active


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Install only the pause-aware active probe around the admitted overlap controller."""
    for name in ("COORDINATION_FREEZE.json", "CONTROLLER_PAUSE_SAFE_FREEZE.json"):
        verify_sources(read(output / name))
    frozen_active = controller_overlap.active
    controller_overlap.active = active_for(output, frozen_active)
    try:
        controller_overlap.run(output, raw_root, teacher_path)
    finally:
        controller_overlap.active = frozen_active


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease((args.output.resolve() / "controller").resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())
