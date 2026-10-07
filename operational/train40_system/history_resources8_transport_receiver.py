"""Run frozen H8 transport extraction with a continuously drained monitor pipe."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.durable_io import atomic_json


def verify_admission(output: Path) -> dict[str, Any]:
    """Verify the receiver freeze and its immutable transport predecessor."""

    from operational.train40_system.contracts import read, verify_sources

    freeze = read(output / "H8_RECEIVER_FREEZE.json")
    verify_sources(freeze)
    directory = ROOT / "operational/train40_system"
    for field, path in (
        ("entrypoint_sha256", Path(__file__)),
        ("receiver_source_sha256", directory / "resource_monitor_receiver.py"),
        ("transport_freeze_sha256", output / "H8_TRANSPORT_FREEZE.json"),
        (
            "receiver_live_admission_sha256",
            output / "h8_transport_admission/H8_RECEIVER_LIVE_ADMISSION.json",
        ),
    ):
        if freeze[field] != digest(path):
            raise ValueError(f"H8 receiver contract changed: {path.name}")
    transport = read(output / "H8_TRANSPORT_FREEZE.json")
    verify_sources(transport)
    return freeze


def run(output: Path, raw_root: Path) -> None:
    """Patch only monitor construction and receipt publication around frozen H8."""

    freeze = verify_admission(output)
    freeze_sha = digest(output / "H8_RECEIVER_FREEZE.json")
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system.resource_monitor_receiver import (
        ContinuousC2FResourceMonitor,
    )

    original_monitor = base.C2FResourceMonitor
    original_kernel_atomic = kernel.atomic_json
    original_base_atomic = base.atomic_json
    original_transport_atomic = transport.atomic_json
    holder: dict[str, ContinuousC2FResourceMonitor] = {}
    status = "FAILED"

    class BoundReceiverMonitor(ContinuousC2FResourceMonitor):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            holder["monitor"] = self

    def annotate(path: Path, value: dict[str, Any]) -> dict[str, Any]:
        if path.name in {
            "H8_FAST_RUNTIME.json",
            "H8_TRANSPORT_RUNTIME.json",
        } or (path.parent.name == "h8_feature_fragments" and path.name.startswith("query_")):
            result = {**value, "receiver_execution_freeze_sha256": freeze_sha}
            monitor = holder.get("monitor")
            if monitor is not None and path.name.endswith("RUNTIME.json"):
                result["receiver_monitor"] = monitor.snapshot()
            return result
        return value

    def publish_kernel(path: Path, value: dict[str, Any]) -> None:
        original_kernel_atomic(path, annotate(path, value))

    def publish_base(path: Path, value: dict[str, Any]) -> None:
        original_base_atomic(path, annotate(path, value))

    def publish_transport(path: Path, value: dict[str, Any]) -> None:
        original_transport_atomic(path, annotate(path, value))

    base.C2FResourceMonitor = BoundReceiverMonitor
    kernel.atomic_json = publish_kernel
    base.atomic_json = publish_base
    transport.atomic_json = publish_transport
    try:
        transport.run(output, raw_root)
        status = "COMPLETE_OR_PRESERVED"
    finally:
        base.C2FResourceMonitor = original_monitor
        kernel.atomic_json = original_kernel_atomic
        base.atomic_json = original_base_atomic
        transport.atomic_json = original_transport_atomic
        monitor = holder.get("monitor")
        atomic_json(
            output / "H8_RECEIVER_RUNTIME.json",
            {
                "status": status,
                "receiver_execution_freeze_sha256": freeze_sha,
                "transport_freeze_sha256": freeze["transport_freeze_sha256"],
                "receiver_monitor": None if monitor is None else monitor.snapshot(),
                "optimizer_updates": 0,
                "checked_utc": datetime.now(UTC).isoformat(),
            },
        )


def main() -> None:
    """Own the unchanged campaign lease and run only admitted H8 extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("H8",), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve())


if __name__ == "__main__":
    main()
