"""Compose CPU-process input preparation and monitoring with frozen TRAIN40 trainers.

Keep module imports lightweight: Windows spawn reimports this entry point in the
monitor process, which must not import Torch or initialize a CUDA context.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import Tensor

    from operational.train40_system.coordination import DeviceTicket
    from operational.train40_system.process_inputs_fast_4 import ProcessInputs


def planned_prefetch_type(base: type) -> type:
    """Add deterministic order hints without changing frozen device-ticket handling."""

    class PlannedPrefetch(base):
        def take(self, order: Tensor, start: int, batch_size: int) -> DeviceTicket:
            if order is not self.order:
                if self.pending:
                    raise ValueError("Previous epoch still has unconsumed batches")
                if self.order is not None and (
                    self.expected != len(self.order) or self.scheduled != len(self.order)
                ):
                    raise ValueError("Previous epoch was not completely consumed")
            self.source.prepare_order(order, start, batch_size)
            return super().take(order, start, batch_size)

    return PlannedPrefetch


def verify_admission(output: Path) -> dict:
    """Require byte-bound source, parity and predecessor admission receipts."""
    from operational.efficient_context.common import digest
    from operational.train40_system.contracts import read, verify_sources

    freeze = read(output / "HOST_FAST_4_FREEZE.json")
    verify_sources(freeze)
    if freeze["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Host fast 4 engine differs from engineering admission")
    for name, expected in freeze["predecessor_freezes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Host fast 4 predecessor differs: {name}")
        verify_sources(read(output / name))
    for name, expected in freeze["QA"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Host fast 4 QA receipt differs: {name}")
    return freeze


def run(output: Path, arm: str) -> None:
    """Delegate unchanged scientific loops, restoring all process-local bindings."""
    if arm not in {"a5", "c2f"}:
        raise ValueError("Only existing A5/C2F TRAIN40 arms are admitted")
    verify_admission(output)

    from operational.efficient_context.common import digest
    from operational.train40_system import (
        engine_graph_replay,
        engine_graph_replay_scalar_safe,
        engine_overlap,
    )
    from operational.train40_system.checkpoint import DurableState
    from operational.train40_system.durable_io import atomic_json
    from operational.train40_system.process_inputs_fast_4 import ProcessInputs
    from operational.train40_system.resource_monitor import ResourceMonitor

    target = engine_graph_replay if arm == "a5" else engine_overlap
    names = ("SealedInputs", "DevicePrefetch", "resource_guard")
    originals = {name: getattr(target, name) for name in names}
    original_scan = getattr(target, "scan_snapshot", None)
    original_save = cast(Any, DurableState.save)
    sources: list[ProcessInputs] = []
    monitor = ResourceMonitor(output)
    directory = output / "fits" / f"{arm}_seed7"

    def inputs(path: Path) -> ProcessInputs:
        source = ProcessInputs(path)
        sources.append(source)
        return source

    def counters() -> dict:
        return {
            "schema": "train40_host_fast_4_runtime_v1",
            "freeze_sha256": digest(output / "HOST_FAST_4_FREEZE.json"),
            "monitor": monitor.snapshot(),
            "inputs": [source.snapshot() for source in sources],
            "checkpoint_contract_unchanged": True,
            "sampler_loss_optimizer_RNG_and_precision_unchanged": True,
        }

    def save(self: DurableState, *args: object, **kwargs: object) -> None:
        original_save(self, *args, **kwargs)
        atomic_json(
            directory / "HOST_FAST_4_COUNTERS.json",
            {
                **counters(),
                "committed_updates": self.committed,
                "durable_updates": self.durable,
            },
        )

    status = "STARTING"
    try:
        monitor.start()
        target.SealedInputs = cast(Any, inputs)
        target.DevicePrefetch = cast(Any, planned_prefetch_type(originals["DevicePrefetch"]))
        target.resource_guard = monitor.guard
        if original_scan is not None:
            cast(Any, target).scan_snapshot = monitor.snapshot
        DurableState.save = cast(Any, save)
        atomic_json(directory / "HOST_FAST_4_RUNTIME.json", {**counters(), "status": status})
        if arm == "a5":
            engine_graph_replay_scalar_safe.run(output, arm)
        else:
            engine_overlap.run(output, arm)
        status = "DELEGATE_RETURNED"
    except BaseException:
        status = "FAILED"
        raise
    finally:
        for name, original in originals.items():
            setattr(target, name, original)
        if original_scan is not None:
            cast(Any, target).scan_snapshot = original_scan
        DurableState.save = original_save
        try:
            for source in sources:
                source.close()
        finally:
            monitor.close()
        atomic_json(directory / "HOST_FAST_4_RUNTIME.json", {**counters(), "status": status})


def main() -> None:
    """Parse one admitted producer task and authenticate the existing writer lease."""
    from operational.efficient_context.common import Lease
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)


if __name__ == "__main__":
    main()
