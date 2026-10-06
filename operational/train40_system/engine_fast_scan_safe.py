"""Restore fast-scan monkey patches even when its early sidecar write fails."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from operational.efficient_context.common import Lease, digest
from operational.train40_system import engine_fast_scan, engine_overlap
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT


def run(output: Path, arm: str) -> None:
    """Delegate to the frozen fast-scan engine with unconditional restoration."""

    freeze = read(output / "GRAPH_REPLAY_FREEZE.json")
    verify_sources(freeze)
    if freeze["fast_scan_safe_engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Safe fast-scan engine differs from graph-replay admission")

    original_guard = engine_overlap.resource_guard
    original_save = cast(Any, DurableState.save)
    try:
        engine_fast_scan.run(output, arm)
    finally:
        engine_overlap.resource_guard = original_guard
        DurableState.save = original_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
