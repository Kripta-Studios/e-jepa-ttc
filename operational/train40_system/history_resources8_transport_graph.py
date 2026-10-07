"""Run the admitted H8 extractor with exact-byte transport CUDA-graph replay."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.durable_io import atomic_json


def verify_admission(output: Path) -> dict[str, Any]:
    """Verify the additive transport freeze and every immediate predecessor."""

    from operational.train40_system.contracts import read, verify_sources

    freeze = read(output / "H8_TRANSPORT_FREEZE.json")
    verify_sources(freeze)
    directory = ROOT / "operational/train40_system"
    for field, path in (
        ("entrypoint_sha256", Path(__file__)),
        ("transport_graph_sha256", directory / "h8_transport_graph.py"),
        ("base_freeze_sha256", output / "H8_FAST_FREEZE.json"),
        (
            "admission_sha256",
            output / "h8_transport_admission/H8_TRANSPORT_ADMISSION.json",
        ),
        (
            "bounded_raw_admission_sha256",
            output / "h8_transport_admission/H8_BOUNDED_RAW_ADMISSION.json",
        ),
        ("bounded_raw_source_sha256", directory / "h8_bounded_union.py"),
    ):
        if freeze[field] != digest(path):
            raise ValueError(f"H8 transport contract changed: {path.name}")
    admission = read(output / "h8_transport_admission/H8_TRANSPORT_ADMISSION.json")
    if admission["status"] != "PASSED":
        raise ValueError("Real H8 transport admission is required")
    bounded = read(output / "h8_transport_admission/H8_BOUNDED_RAW_ADMISSION.json")
    if (
        bounded["status"] != "PASSED"
        or not bounded["all_tensors_exact_bytes"]
        or int(bounded["oversized_union_cases"]) < 2
        or int(bounded["retained_bytes_max"]) > 256 * 1024**2
    ):
        raise ValueError("Real bounded H8 raw parity is required")
    return freeze


def run(output: Path, raw_root: Path) -> None:
    """Patch only transport aliases and bind new fragment receipts durably."""

    freeze = verify_admission(output)
    freeze_sha = digest(output / "H8_TRANSPORT_FREEZE.json")
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system.h8_bounded_union import prepare as bounded_prepare
    from operational.train40_system.h8_transport_graph import H8TransportGraph

    original_atomic = kernel.atomic_json
    original_base_atomic = base.atomic_json
    original_prepare = kernel.prepare
    graph = H8TransportGraph(
        max_entries=int(freeze["graph_cache_entries"]),
        warmup_calls=int(freeze["graph_warmup_calls"]),
    )
    status = "FAILED"

    def publish(path: Path, value: dict[str, Any]) -> None:
        if path.parent.name == "h8_feature_fragments" and path.name.startswith("query_"):
            value = {**value, "transport_execution_freeze_sha256": freeze_sha}
        original_atomic(path, value)

    def publish_base(path: Path, value: dict[str, Any]) -> None:
        if path.name == "H8_FAST_RUNTIME.json":
            value = {
                **value,
                "transport_execution_freeze_sha256": freeze_sha,
                "transport_graph": graph.snapshot(),
            }
        original_base_atomic(path, value)

    kernel.atomic_json = publish
    kernel.prepare = bounded_prepare
    base.atomic_json = publish_base
    try:
        with graph.install():
            base.run(output, raw_root)
        status = "COMPLETE_OR_PRESERVED"
    finally:
        kernel.atomic_json = original_atomic
        kernel.prepare = original_prepare
        base.atomic_json = original_base_atomic
        graph.close()
        snapshot = graph.snapshot()
        atomic_json(
            output / "H8_TRANSPORT_RUNTIME.json",
            {
                "status": status,
                "transport_execution_freeze_sha256": freeze_sha,
                "base_freeze_sha256": freeze["base_freeze_sha256"],
                "graph": snapshot,
                "optimizer_updates": 0,
                "checked_utc": datetime.now(UTC).isoformat(),
            },
        )


def main() -> None:
    """Own the unchanged campaign lease and execute only the admitted H8 task."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("H8",), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve())


if __name__ == "__main__":
    main()
