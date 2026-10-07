"""Run admitted H8 extraction with four raw workers and eight transport arenas."""

# ruff: noqa: B009, B010 -- heterogeneous module bindings are restored dynamically.

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, Lease, digest

RAW_WORKER_COUNT = 4
ARENA_COUNT = 8
MEMORY_PRESSURE_BYTES = 23_000_000_000


@contextmanager
def four_raw_workers() -> Iterator[None]:
    """Limit the executor only; retain eight arena leases and pending submissions."""
    from operational.train40_system import h8_shared_pool

    original = h8_shared_pool.ProcessPoolExecutor

    def executor(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        requested = kwargs.get("max_workers", args[0] if args else None)
        if requested != ARENA_COUNT:
            raise ValueError("H8 four-worker route requires eight transport arenas")
        if args:
            args = (RAW_WORKER_COUNT, *args[1:])
        else:
            kwargs = {**kwargs, "max_workers": RAW_WORKER_COUNT}
        return original(*args, **kwargs)

    h8_shared_pool.ProcessPoolExecutor = executor
    try:
        yield
    finally:
        h8_shared_pool.ProcessPoolExecutor = original


def verify_admission(output: Path) -> dict[str, Any]:
    """Verify this additive policy and its admitted memory predecessor."""
    from operational.train40_system.contracts import read, verify_sources
    from operational.train40_system.history_resources8_transport_memory import (
        verify_admission as predecessor,
    )

    freeze = read(output / "H8_FOUR_FREEZE.json")
    verify_sources(freeze)
    if freeze["entrypoint_sha256"] != digest(Path(__file__)):
        raise ValueError("H8 four-worker entrypoint changed")
    if freeze["memory_freeze_sha256"] != digest(output / "H8_MEMORY_FREEZE.json"):
        raise ValueError("H8 four-worker predecessor changed")
    if (
        freeze["raw_worker_count"] != RAW_WORKER_COUNT
        or freeze["arena_count"] != ARENA_COUNT
        or freeze["memory_pressure_bytes"] != MEMORY_PRESSURE_BYTES
        or freeze["host_reserve_bytes"] != 2 * 1024**3
    ):
        raise ValueError("H8 four-worker execution bounds changed")
    for name, sha256 in freeze["QA"].items():
        if digest(output / name) != sha256:
            raise ValueError("H8 four-worker QA receipt changed")
    predecessor(output)
    return freeze


def run(output: Path, raw_root: Path) -> None:
    """Compose bounded workers with 23 GB throttling and live memory recovery."""
    verify_admission(output)
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system import history_resources8_transport_memory as memory
    from operational.train40_system import history_resources8_transport_receiver as receiver
    from operational.train40_system import resource_monitor, resource_monitor_receiver
    from operational.train40_system.h8_memory_flow import MemoryFlowMonitor

    if memory.TREE_LIMIT != 20_000_000_000:
        raise ValueError("H8 aggregate RSS policy changed")
    if resource_monitor.HOST_AVAILABLE_MIN_BYTES != 2 * 1024**3:
        raise ValueError("H8 physical host reserve changed")
    freeze_sha = digest(output / "H8_FOUR_FREEZE.json")
    modules = (kernel, base, transport, receiver)
    originals = tuple(getattr(module, "atomic_json") for module in modules)
    original_allowance = memory.memory_allowance
    original_receiver = resource_monitor_receiver.ContinuousC2FResourceMonitor

    @contextmanager
    def pressure_allowance() -> Iterator[None]:
        with original_allowance():
            previous = resource_monitor.TREE_RSS_LIMIT_BYTES
            resource_monitor.TREE_RSS_LIMIT_BYTES = MEMORY_PRESSURE_BYTES
            try:
                yield
            finally:
                resource_monitor.TREE_RSS_LIMIT_BYTES = previous

    def annotate(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        if path.name.endswith("RUNTIME.json") or path.parent.name == "h8_feature_fragments":
            return {
                **value,
                "four_worker_execution_freeze_sha256": freeze_sha,
                "raw_worker_count": RAW_WORKER_COUNT,
                "arena_count": ARENA_COUNT,
                "tree_rss_limit_bytes": MEMORY_PRESSURE_BYTES,
                "memory_policy": "throttle_RSS_wait_host_reserve_without_RAM_exit",
            }
        return value

    def publisher(original: Any) -> Any:  # noqa: ANN401
        def publish(path: Path, value: dict[str, Any]) -> None:
            original(path, annotate(path, value))

        return publish

    for module, original in zip(modules, originals, strict=True):
        setattr(module, "atomic_json", publisher(original))
    memory.memory_allowance = pressure_allowance
    resource_monitor_receiver.ContinuousC2FResourceMonitor = MemoryFlowMonitor
    try:
        with four_raw_workers():
            memory.run(output, raw_root)
    finally:
        memory.memory_allowance = original_allowance
        resource_monitor_receiver.ContinuousC2FResourceMonitor = original_receiver
        for module, original in zip(modules, originals, strict=True):
            setattr(module, "atomic_json", original)


def main() -> None:
    """Acquire the existing single-writer lease before resuming H8."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("H8",), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve())


if __name__ == "__main__":
    main()


__all__ = ["ARENA_COUNT", "RAW_WORKER_COUNT", "four_raw_workers", "run"]
