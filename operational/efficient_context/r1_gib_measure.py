"""Three paired R1 blocks with one GiB per route and unchanged model dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import r1_measure, r1_profile
from .common import ROOT, atomic_json, digest


def main() -> int:
    """Reuse the frozen profiler and startup audit with explicit1GiB capacity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/efficient_context_20261004/r1_20261008/resident_1g",
    )
    output = parser.parse_args().output.resolve()
    atomic_json(
        output / "measurement/CAPACITY_ADMISSION.json",
        dict(
            capacity_bytes_per_route=1024**3,
            combined_resident_capacity_bytes=2 * 1024**3,
            wrapper_sha256=digest(Path(__file__)),
            optimizer_updates=0,
            selection="one production capacity selected from prior measured max971.02MiB; no sweep",
        ),
    )
    original = r1_profile.run

    def run_gib(args: argparse.Namespace) -> None:
        args.capacity_mib = 1024
        original(args)

    r1_profile.run = run_gib
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], "--output", str(output)]
    try:
        return r1_measure.main()
    finally:
        sys.argv = previous_argv
        r1_profile.run = original


if __name__ == "__main__":
    raise SystemExit(main())
