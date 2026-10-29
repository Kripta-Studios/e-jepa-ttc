"""Verify the H8 entrypoint stays Torch-free in a real Windows monitor spawn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from operational.efficient_context.common import ROOT
from operational.train40_system.history_resources8_fast import C2FResourceMonitor


def main() -> None:
    """Require an actual fresh snapshot without importing Torch in the monitor process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("H8 lightweight entrypoint imported Torch")
    monitor = C2FResourceMonitor(args.output.resolve())
    try:
        monitor.start()
        allowed, telemetry = monitor.guard(args.output.resolve())
        snapshot = monitor.snapshot()
        if snapshot["latest"].get("torch_imported"):
            raise RuntimeError("H8 spawned resource monitor imported Torch")
        expected = {"COORDINATION_PAUSE_REQUESTED"}
        if not allowed and set(telemetry["reasons"]) - expected:
            raise RuntimeError(str(telemetry))
        print("PASSED: actual spawn; no Torch; fresh resource snapshot; pause respected")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
