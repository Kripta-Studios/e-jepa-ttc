"""Keep admitted cache training while reducing Windows process-inspection overhead."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

from operational.efficient_context.common import Lease, digest
from operational.train40_system import engine_adaptive_cache, engine_overlap
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def run(output: Path, arm: str) -> None:
    """Delegate all scientific work to the unchanged admitted cache trainer."""
    from operational.train40_system import process_scan

    freeze_path = output / "FAST_PROCESS_SCAN_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
    if freeze["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Fast process scan engine differs from admission")
    if freeze["adaptive_cache_freeze_sha256"] != digest(output / "ADAPTIVE_CACHE_FREEZE.json"):
        raise ValueError("Fast process scan admission differs from its cache baseline")
    original_guard = engine_overlap.resource_guard
    original_save = cast(Any, DurableState.save)
    stats = {"guard_calls": 0, "guard_total_seconds": 0.0}
    directory = output / "fits" / f"{arm}_seed7"

    def guard(path: Path) -> tuple[bool, dict]:
        started = time.perf_counter()
        try:
            return process_scan.resource_guard(path)
        finally:
            stats["guard_calls"] += 1
            stats["guard_total_seconds"] += time.perf_counter() - started

    def save(self: DurableState, *args: object, **kwargs: object) -> None:
        original_save(self, *args, **kwargs)
        atomic_json(
            directory / "FAST_PROCESS_SCAN_COUNTERS.json",
            {
                "committed_updates": self.committed,
                "durable_updates": self.durable,
                **stats,
                "scan": process_scan.snapshot(),
            },
        )

    engine_overlap.resource_guard = guard
    DurableState.save = cast(Any, save)
    atomic_json(
        directory / "FAST_PROCESS_SCAN_RUNTIME.json",
        {
            "freeze_sha256": digest(freeze_path),
            "model_loss_optimizer_sampler_RNG_and_checkpoint_unchanged": True,
            "original_hard_resource_and_update_budget_guards_unchanged": True,
            "heavy_trainer_alias_coverage_extended": True,
        },
    )
    try:
        engine_adaptive_cache.run(output, arm)
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
