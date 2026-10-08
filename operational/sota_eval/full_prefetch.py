"""Process-prefetched execution backend for frozen Garl RGB+event inference.

Only the label-free ``evttc_rgb_transfer.inputs.prepare`` call is moved to two
spawned CPU workers.  The frozen full-Garl runner still owns model loading,
prediction, fragment validation, scientific identity, and sealing.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from operational.efficient_context.common import atomic_json, digest
from operational.sota_eval.prefetch import (
    GIB,
    OrderedPreparePrefetch,
    PrefetchAdmissionTimeoutError,
    PrefetchStoppedError,
    _select_pilot_rows,
    _worker_init,
    assert_prepared_equal,
    pending_rows,
    prepared_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT_OUTPUT = ROOT / "artifacts/sota_campaign_20261008/full_prefetch"
STABLE_PREFETCH = ROOT / "operational/sota_eval/prefetch.py"


class Pool(Protocol):
    def submit(
        self,
        fn: Callable[[Mapping[str, Any]], Any],
        row: Mapping[str, Any],
        /,
    ) -> Future[Any]: ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None: ...


def _full_worker(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    from operational.evttc_rgb_transfer.inputs import prepare

    started_ns = time.perf_counter_ns()
    result = prepare(row)
    prepare_ns = time.perf_counter_ns() - started_ns
    import psutil

    memory = psutil.Process().memory_info()
    return result, {
        "pid": os.getpid(),
        "rss_bytes": int(memory.rss),
        "peak_rss_bytes": int(getattr(memory, "peak_wset", memory.rss)),
        "prepare_ns": prepare_ns,
    }


class FullPrepareExecutor:
    """Route OrderedPreparePrefetch submissions to the full-input worker."""

    def __init__(self, workers: int = 2, pool: Pool | None = None) -> None:
        if workers not in (1, 2):
            raise ValueError("workers must be one or two")
        self._pool = pool or ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(2,),
        )

    def submit(
        self,
        _ignored_event_worker: Callable[[Mapping[str, Any]], Any],
        row: Mapping[str, Any],
        /,
    ) -> Future[Any]:
        return self._pool.submit(_full_worker, row)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


def backend_binding(
    manifest_path: Path, *, workers: int, max_ahead: int, min_available_bytes: int
) -> dict[str, Any]:
    """Bind execution sources without changing the frozen scientific binding."""
    sources = [
        Path(__file__),
        STABLE_PREFETCH,
        ROOT / "operational/evttc_rgb_transfer/inputs.py",
        ROOT / "operational/evttc_rgb_transfer/run.py",
        ROOT / "operational/evttc_rgb_transfer/model.py",
    ]
    return {
        "artifact_type": "full_garl_ordered_spawn_prefetch_backend_v1",
        "manifest_sha256": digest(manifest_path),
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "workers": workers,
        "max_ahead": max_ahead,
        "min_available_bytes": min_available_bytes,
        "worker_torch_threads": 2,
        "consumer_order": "frozen_manifest_order",
        "scientific_binding_changed": False,
        "optimizer_updates": 0,
    }


def _execution_receipt(
    *,
    status: str,
    started_utc: str,
    started: float,
    previous_sha256: str | None,
    adapter: OrderedPreparePrefetch | None,
    error: str | None = None,
) -> dict[str, Any]:
    records = [] if adapter is None else adapter.execution_records
    return {
        "artifact_type": "full_garl_ordered_spawn_prefetch_execution_v1",
        "status": status,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "previous_execution_sha256": previous_sha256,
        "records": records,
        "queries_consumed": len(records),
        "waited_for_ram_seconds": (
            0.0 if adapter is None else adapter.waited_for_ram_seconds
        ),
        "error": error,
        "scientific_binding_changed": False,
        "timing_warning": (
            "Frozen seconds.prepare is consumer wait/retrieval under prefetch. Use "
            "worker_prepare_seconds and backend wall_seconds; overlapped durations are not "
            "additive end-to-end latency."
        ),
    }


def run_prefetched_full(
    output: Path,
    baseline: Path,
    code_root: Path,
    device: str,
    limit: int | None,
    *,
    workers: int = 2,
    max_ahead: int = 2,
) -> None:
    """Run the frozen full-Garl evaluator with a reversible prepare replacement."""
    from operational.evttc_rgb_transfer import inputs
    from operational.evttc_rgb_transfer import run as frozen_run

    manifest_path = baseline / "QUERY_MANIFEST.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"]
    output.mkdir(parents=True, exist_ok=True)
    binding = backend_binding(
        manifest_path, workers=workers, max_ahead=max_ahead, min_available_bytes=2 * GIB
    )
    freeze_path = output / "BACKEND_FREEZE.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("full-prefetch backend freeze differs from execution lineage")
    else:
        atomic_json(freeze_path, binding)
    execution_path = output / "BACKEND_EXECUTION.json"
    previous = digest(execution_path) if execution_path.exists() else None
    started, started_utc = time.perf_counter(), datetime.now(UTC).isoformat()
    if (output / "STOP_REQUEST").exists():
        atomic_json(
            execution_path,
            _execution_receipt(
                status="PAUSED_PRESERVED",
                started_utc=started_utc,
                started=started,
                previous_sha256=previous,
                adapter=None,
                error="stop requested before full-prefetch pool creation",
            ),
        )
        return
    executor = FullPrepareExecutor(workers)
    adapter: OrderedPreparePrefetch | None = None
    try:
        try:
            adapter = OrderedPreparePrefetch(
                pending_rows(output, rows),
                workers=workers,
                max_ahead=max_ahead,
                executor=executor,
                stop_requested=lambda: (output / "STOP_REQUEST").exists(),
            )
        except (PrefetchStoppedError, PrefetchAdmissionTimeoutError) as error:
            status = (
                "PAUSED_PRESERVED"
                if isinstance(error, PrefetchStoppedError)
                else "WAITING_FOR_RAM_TIMEOUT_PRESERVED"
            )
            receipt = _execution_receipt(
                status=status,
                started_utc=started_utc,
                started=started,
                previous_sha256=previous,
                adapter=None,
                error=str(error),
            )
            atomic_json(execution_path, receipt)
            atomic_json(output / "STATE.json", receipt)
            return
        original = inputs.prepare
        inputs.prepare = adapter
        try:
            frozen_run.run(output, baseline, code_root, device, limit)
        finally:
            inputs.prepare = original
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if adapter is not None:
            atomic_json(
                execution_path,
                _execution_receipt(
                    status="FINISHED_OR_PRESERVED_BY_FROZEN_RUNNER",
                    started_utc=started_utc,
                    started=started,
                    previous_sha256=previous,
                    adapter=adapter,
                ),
            )


def pilot(
    manifest_path: Path,
    output: Path,
    *,
    count: int = 32,
    selection: str = "first-per-sequence",
) -> dict[str, Any]:
    """Compare process and direct full-input preparation without model execution."""
    import psutil

    from operational.evttc_rgb_transfer.inputs import prepare

    all_rows = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"]
    rows = _select_pilot_rows(all_rows, count, selection)
    if len(rows) != count:
        raise ValueError(f"pilot selected {len(rows)} of {count} required rows")
    parent = psutil.Process()
    executor = FullPrepareExecutor(2)
    begin = time.perf_counter()
    adapter = OrderedPreparePrefetch(rows, workers=2, max_ahead=2, executor=executor)
    process_outputs: list[dict[str, Any]] = []
    completion: list[float] = []
    try:
        for row in rows:
            process_outputs.append(adapter(row))
            completion.append(time.perf_counter() - begin)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    process_seconds = time.perf_counter() - begin
    process_parent_rss = int(parent.memory_info().rss)

    direct_outputs: list[dict[str, Any]] = []
    direct_query_seconds: list[float] = []
    begin = time.perf_counter()
    for row in rows:
        query_started = time.perf_counter()
        direct_outputs.append(prepare(row))
        direct_query_seconds.append(time.perf_counter() - query_started)
    direct_seconds = time.perf_counter() - begin
    records = []
    for row, process_output, direct_output in zip(
        rows, process_outputs, direct_outputs, strict=True
    ):
        assert_prepared_equal(process_output, direct_output)
        records.append(
            {
                "query_id": row["query_id"],
                "process_sha256": prepared_sha256(process_output),
                "direct_sha256": prepared_sha256(direct_output),
                "bit_exact": True,
            }
        )
    process_after_first = completion[-1] - completion[0]
    direct_after_first = sum(direct_query_seconds[1:])
    result = {
        "artifact_type": "full_garl_prepare_prefetch_pilot_v1",
        "status": "PASSED",
        "decision": (
            "QUALIFIED_ENABLE_ON_NEXT_RESUME"
            if direct_after_first > process_after_first
            else "DO_NOT_ENABLE_PROCESS_OVERHEAD_DOMINATES"
        ),
        "queries": count,
        "selection": selection,
        "manifest_sha256": digest(manifest_path),
        "sources": backend_binding(
            manifest_path, workers=2, max_ahead=2, min_available_bytes=2 * GIB
        )["sources"],
        "workers": 2,
        "max_ahead": 2,
        "process_seconds_including_spawn": process_seconds,
        "process_completion_seconds_including_spawn": completion,
        "process_seconds_after_first_completion": process_after_first,
        "direct_seconds_after_process_run": direct_seconds,
        "direct_query_seconds": direct_query_seconds,
        "steady_after_first_speedup": direct_after_first / process_after_first,
        "parent_rss_bytes_after_process": process_parent_rss,
        "parent_rss_bytes_after_direct": int(parent.memory_info().rss),
        "worker_memory": adapter.worker_memory,
        "max_worker_peak_rss_bytes": max(
            item["peak_rss_bytes"] for item in adapter.worker_memory
        ),
        "waited_for_ram_seconds": adapter.waited_for_ram_seconds,
        "array_payload_bytes_per_query": [
            sum(value.nbytes for value in prepared.values() if isinstance(value, np.ndarray))
            for prepared in process_outputs
        ],
        "records": records,
        "labels_read": False,
        "models_run": False,
        "optimizer_updates": 0,
        "interpretation": (
            "Process timing includes spawn/import and direct timing runs second with possible OS "
            "cache benefit. Deployment benefit depends on overlap with frozen GPU inference."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "PILOT.json", result)
    atomic_json(
        output / "BACKEND_FREEZE.json",
        backend_binding(
            manifest_path, workers=2, max_ahead=2, min_available_bytes=2 * GIB
        ),
    )
    atomic_json(
        output / "SHA256.json",
        {
            name: digest(output / name)
            for name in ("PILOT.json", "BACKEND_FREEZE.json")
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--manifest", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    pilot_parser.add_argument("--count", type=int, default=32)
    pilot_parser.add_argument(
        "--selection", choices=("sequential", "first-per-sequence"),
        default="first-per-sequence",
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--baseline", type=Path, required=True)
    run_parser.add_argument("--code-root", type=Path, required=True)
    run_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    run_parser.add_argument("--max-ahead", choices=(1, 2), type=int, default=2)
    args = parser.parse_args()
    if args.command == "pilot":
        print(
            json.dumps(
                pilot(
                    args.manifest.resolve(),
                    args.output.resolve(),
                    count=args.count,
                    selection=args.selection,
                ),
                indent=2,
            )
        )
    else:
        run_prefetched_full(
            args.output.resolve(),
            args.baseline.resolve(),
            args.code_root.resolve(),
            args.device,
            args.limit,
            workers=args.workers,
            max_ahead=args.max_ahead,
        )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
