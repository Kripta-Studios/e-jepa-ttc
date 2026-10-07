"""Resume the frozen H8 extraction with admitted host/GPU execution optimizations."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine_c2f_graph_replay import C2FResourceMonitor

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from operational.train40_system.h8_shared_pool import SharedArrayProcessPool


def verify_admission(output: Path) -> dict[str, Any]:
    """Bind the optimized execution to the original fragments and real parity evidence."""
    from operational.train40_system.contracts import read, verify_sources

    freeze = read(output / "H8_FAST_FREEZE.json")
    verify_sources(freeze)
    if freeze["entrypoint_sha256"] != digest(Path(__file__)):
        raise ValueError("Optimized H8 entrypoint differs from its freeze")
    for field, relative in (
        ("baseline_binding_sha256", "h8_feature_fragments/BINDING.json"),
        ("protocol_sha256", "TRAINING_PROTOCOL.json"),
        ("admission_sha256", "h8_fast_admission/ADMISSION.json"),
    ):
        if freeze[field] != digest(output / relative):
            raise ValueError("Optimized H8 contract changed: " + relative)
    if read(output / "h8_fast_admission/ADMISSION.json")["status"] != "PASSED":
        raise ValueError("Real H8 tensor and feature parity required")
    if digest(Path(freeze["backend_source"])) != freeze["backend_source_sha256"]:
        raise ValueError("Installed H8 CUDA graph backend changed")
    return freeze


def run(output: Path, raw_root: Path) -> None:
    """Retain canonical data, slot order, FP32 operations and resumable receipt filenames."""
    freeze = verify_admission(output)
    freeze_sha = digest(output / "H8_FAST_FREEZE.json")
    monitor = C2FResourceMonitor(output)
    monitor.start()
    try:
        import torch

        from operational.simplex_t_shared_route import adapter
        from operational.train40_system import history_resources8 as kernel
        from operational.train40_system.contracts import read
        from operational.train40_system.h8_fast_extract import H8Extractor

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        original_guard, original_atomic, original_extract = (
            kernel.resource_guard,
            kernel.atomic_json,
            adapter.extract,
        )
        original_pool = kernel.ProcessPoolExecutor
        extractor: H8Extractor | None = None
        calls, guard_seconds, extract_seconds, progress_seconds = 0, 0.0, 0.0, 0.0
        last_progress, last_snapshot = 0.0, 0.0
        last_row = 0
        shared_pool: Any = None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        segment = output / "h8_fast_execution" / f"SEGMENT_{stamp}.json"
        metadata = {
            "freeze_sha256": freeze_sha,
            "baseline_binding_sha256": freeze["baseline_binding_sha256"],
            "mode": freeze["mode"],
            "shared_memory": freeze["shared_memory"],
            "optimizer_updates": 0,
            "started_utc": datetime.now(UTC).isoformat(),
            "first_new_row": None,
        }

        def snapshot(status: str) -> None:
            atomic_json(
                output / "H8_FAST_RUNTIME.json",
                {
                    **metadata,
                    "status": status,
                    "processed_rows": last_row,
                    "extract_calls": calls,
                    "guard_seconds": guard_seconds,
                    "extract_seconds": extract_seconds,
                    "progress_publication_seconds": progress_seconds,
                    "extractor": None if extractor is None else extractor.snapshot(),
                    "monitor": monitor.snapshot(),
                    "shared_pool": None if shared_pool is None else shared_pool.snapshot(),
                    "checked_utc": datetime.now(UTC).isoformat(),
                },
            )

        def guard(path: Path) -> tuple[bool, dict[str, Any]]:
            nonlocal guard_seconds
            started = time.perf_counter()
            result = monitor.guard(path)
            guard_seconds += time.perf_counter() - started
            return result

        def extract(model: str, models: dict, events: Tensor, delta: Tensor) -> np.ndarray:
            nonlocal extractor, calls, extract_seconds
            if extractor is None:
                extractor = H8Extractor(models, mode=freeze["mode"])
                metadata["first_new_row"] = last_row
                atomic_json(segment, metadata)
            started = time.perf_counter()
            result = extractor(model, models, events, delta)
            extract_seconds += time.perf_counter() - started
            calls += 1
            return result

        def publish(path: Path, value: dict) -> None:
            nonlocal last_progress, last_row, last_snapshot, progress_seconds
            now = time.monotonic()
            if path.name == "H8_FEATURE_PROGRESS.json" and value.get("status") == "RUNNING":
                last_row = int(value["completed_rows"])
                if last_row % 16 and now - last_progress < 5:
                    return
                begun = time.perf_counter()
                original_atomic(path, value)
                progress_seconds += time.perf_counter() - begun
                last_progress = now
                if now - last_snapshot >= 30:
                    snapshot("RUNNING")
                    last_snapshot = now
                return
            if path.parent.name == "h8_feature_fragments" and path.name.startswith("query_"):
                value = {**value, "execution_freeze_sha256": freeze_sha}
            original_atomic(path, value)

        def pool_factory(
            *, max_workers: int, initializer: Callable[[], None]
        ) -> SharedArrayProcessPool:
            nonlocal shared_pool
            from operational.train40_system.h8_shared_pool import h8_shared_pool

            shared_pool = h8_shared_pool(
                max_workers=max_workers, initializer=initializer, profile=True
            )
            return shared_pool

        kernel.resource_guard = guard
        kernel.atomic_json = publish
        adapter.extract = extract
        if freeze["shared_memory"]:
            kernel.ProcessPoolExecutor = cast(Any, pool_factory)
        try:
            snapshot("STARTING")
            kernel.history_cache(output, raw_root)
            manifest = output / "H8_FEATURE_MANIFEST.json"
            status = (
                "COMPLETE"
                if manifest.exists() and read(manifest)["status"] == "COMPLETE_VERIFIED"
                else "PAUSED_PRESERVED"
            )
            original_atomic(
                output / "H8_FEATURE_PROGRESS.json",
                {"status": status, "completed_rows": last_row, "total_rows": 88744},
            )
            atomic_json(
                segment,
                {**metadata, "status": status, "new_fragments": calls, "last_row": last_row},
            )
            snapshot(status)
        finally:
            kernel.resource_guard = original_guard
            kernel.atomic_json = original_atomic
            kernel.ProcessPoolExecutor = original_pool
            adapter.extract = original_extract
    finally:
        monitor.close()


def main() -> None:
    """Own the original campaign writer lease; never train or change frozen weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("H8",), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve())


if __name__ == "__main__":
    main()
