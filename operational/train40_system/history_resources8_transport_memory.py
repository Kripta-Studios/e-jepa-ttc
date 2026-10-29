"""Admit a bounded aggregate RSS allowance for shared-memory H8 extraction."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system import resource_monitor

TREE_LIMIT = 20_000_000_000


@contextmanager
def memory_allowance() -> Iterator[None]:
    """Raise only process-tree RSS accounting; retain the physical host reserve."""
    previous = resource_monitor.TREE_RSS_LIMIT_BYTES
    if resource_monitor.HOST_AVAILABLE_MIN_BYTES != 2 * 1024**3:
        raise ValueError("The physical host reserve must remain 2 GiB")
    resource_monitor.TREE_RSS_LIMIT_BYTES = TREE_LIMIT
    try:
        yield
    finally:
        resource_monitor.TREE_RSS_LIMIT_BYTES = previous


def verify_admission(output: Path) -> dict[str, Any]:
    """Bind the memory policy to its tested source and frozen predecessor."""
    from operational.train40_system.contracts import read, verify_sources
    from operational.train40_system.history_resources8_transport_receiver import (
        verify_admission as predecessor,
    )

    freeze = read(output / "H8_MEMORY_FREEZE.json")
    verify_sources(freeze)
    if freeze["receiver_freeze_sha256"] != digest(output / "H8_RECEIVER_FREEZE.json"):
        raise ValueError("H8 memory predecessor changed")
    if freeze["tree_limit_bytes"] != TREE_LIMIT or freeze["host_reserve_bytes"] != 2 * 1024**3:
        raise ValueError("H8 memory limits changed")
    for name, sha in freeze["QA"].items():
        if digest(output / name) != sha:
            raise ValueError("H8 memory QA receipt changed")
    predecessor(output)
    return freeze


def run(output: Path, raw_root: Path) -> None:
    """Keep frozen inputs, extraction, resume checks, and scientific budgets."""
    verify_admission(output)
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_receiver as receiver
    from operational.train40_system.h8_resume_direct import history_cache

    freeze_sha = digest(output / "H8_MEMORY_FREEZE.json")
    old_kernel, old_base = kernel.atomic_json, base.atomic_json
    old_history = kernel.history_cache

    def annotate(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        if path.name == "H8_FAST_RUNTIME.json" or path.parent.name == "h8_feature_fragments":
            return {
                **value,
                "memory_execution_freeze_sha256": freeze_sha,
                "tree_rss_limit_bytes": TREE_LIMIT,
                "host_available_min_bytes": 2 * 1024**3,
            }
        return value

    def publish_kernel(path: Path, value: dict[str, Any]) -> None:
        old_kernel(path, annotate(path, value))

    def publish_base(path: Path, value: dict[str, Any]) -> None:
        old_base(path, annotate(path, value))

    kernel.atomic_json, base.atomic_json = publish_kernel, publish_base
    kernel.history_cache = history_cache
    try:
        with memory_allowance():
            receiver.run(output, raw_root)
    finally:
        kernel.atomic_json, base.atomic_json = old_kernel, old_base
        kernel.history_cache = old_history


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
