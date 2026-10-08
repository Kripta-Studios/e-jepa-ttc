"""Real CPU admission with one GiB per resident sensor buffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import r1_parity
from .common import ROOT, atomic_json, digest
from .r1_reader import ChunkSource, ResidentReplay


class ResidentGiB(ResidentReplay):
    """Fit the measured 971 MiB maximum window with a fixed one-GiB budget."""

    def __init__(self, source: ChunkSource) -> None:
        super().__init__(source, capacity_bytes=1024**3)


def main() -> int:
    """Run unchanged canonical tensor parity with the selected resident capacity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/efficient_context_20261004/r1_20261008/resident_1g/window_parity",
    )
    output = parser.parse_args().output.resolve()
    atomic_json(
        output / "CAPACITY_ADMISSION.json",
        dict(
            capacity_bytes=1024**3,
            wrapper_sha256=digest(Path(__file__)),
            selection="one value exceeding measured TRAIN-window maximum971.02MiB; no sweep",
            optimizer_updates=0,
        ),
    )
    original = r1_parity.ResidentReplay
    r1_parity.ResidentReplay = ResidentGiB
    try:
        r1_parity.run(output)
    finally:
        r1_parity.ResidentReplay = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
