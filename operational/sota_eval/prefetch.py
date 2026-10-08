"""Ordered process prefetch for the frozen label-free EvTTC preparation path.

This module changes only execution scheduling.  Workers call the frozen
``operational.evttc_transfer.inputs.prepare`` function and the consumer emits
results in the original manifest order.  Models, tensors, precision, queries,
and prediction receipts remain owned by the frozen runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from operational.efficient_context.common import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/sota_campaign_20261008/prefetch"
GIB = 1024**3


class Executor(Protocol):
    """Small executor contract used to make ordered scheduling testable."""

    def submit(
        self,
        fn: Callable[[Mapping[str, Any]], Any],
        row: Mapping[str, Any],
        /,
    ) -> Future[Any]: ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None: ...


class PrefetchStoppedError(InterruptedError):
    """Raised when admission observes the frozen runner's stop request."""


class PrefetchAdmissionTimeoutError(TimeoutError):
    """Raised when an empty queue cannot safely admit work before its deadline."""


def _worker_init(torch_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    import torch

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)


def _worker_prepare(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    from operational.evttc_transfer.inputs import prepare

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


def prepared_sha256(prepared: Mapping[str, Any]) -> str:
    """Hash every prepared value with explicit array dtype and shape identities."""
    checksum = hashlib.sha256()
    for key in sorted(prepared):
        checksum.update(key.encode("utf-8") + b"\0")
        value = prepared[key]
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            checksum.update(contiguous.dtype.str.encode("ascii") + b"\0")
            checksum.update(json.dumps(contiguous.shape).encode("ascii") + b"\0")
            checksum.update(contiguous.tobytes())
        elif isinstance(value, np.generic):
            scalar = np.asarray(value)
            checksum.update(scalar.dtype.str.encode("ascii") + b"\0")
            checksum.update(scalar.tobytes())
        else:
            checksum.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return checksum.hexdigest()


def assert_prepared_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Require bit-exact arrays and equal non-array values."""
    if set(left) != set(right):
        raise ValueError("prepared mappings expose different keys")
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (np.ndarray, np.generic)) or isinstance(b, (np.ndarray, np.generic)):
            aa, bb = np.asarray(a), np.asarray(b)
            if aa.dtype != bb.dtype or aa.shape != bb.shape or not np.array_equal(aa, bb):
                raise ValueError(f"prepared value differs bitwise: {key}")
        elif a != b:
            raise ValueError(f"prepared value differs: {key}")


class OrderedPreparePrefetch:
    """Callable ordered adapter with a bounded number of process futures."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        workers: int = 2,
        max_ahead: int = 2,
        min_available_bytes: int = 2 * GIB,
        torch_threads: int = 2,
        executor: Executor | None = None,
        available_bytes: Callable[[], int] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        wait_seconds: float = 2.0,
        admission_timeout_seconds: float = 300.0,
        hash_limit: int = 32,
    ) -> None:
        if workers not in (1, 2):
            raise ValueError("workers must be one or two")
        if not 1 <= max_ahead <= 2:
            raise ValueError("max_ahead must be one or two")
        self._rows = list(rows)
        self._max_ahead = min(max_ahead, len(self._rows))
        self._min_available = min_available_bytes
        self._available = available_bytes or self._system_available
        self._stop_requested = stop_requested or (lambda: False)
        self._wait_seconds = wait_seconds
        self._admission_timeout_seconds = admission_timeout_seconds
        self._hash_limit = hash_limit
        self._next_consume = 0
        self._next_submit = 0
        self._futures: dict[int, Future[Any]] = {}
        self.worker_memory: list[dict[str, int]] = []
        self.execution_records: list[dict[str, Any]] = []
        self.waited_for_ram_seconds = 0.0
        self._owns_executor = executor is None
        self._executor = executor or ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(torch_threads,),
        )
        try:
            self._fill(block_if_empty=True)
        except BaseException:
            if self._owns_executor:
                self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    @staticmethod
    def _system_available() -> int:
        import psutil

        return int(psutil.virtual_memory().available)

    def _wait_for_headroom(self, *, block: bool) -> bool:
        if self._available() >= self._min_available:
            return True
        if not block:
            return False
        started = time.monotonic()
        while self._available() < self._min_available:
            if self._stop_requested():
                raise PrefetchStoppedError(
                    "stop requested while waiting for prefetch RAM headroom"
                )
            if time.monotonic() - started >= self._admission_timeout_seconds:
                raise PrefetchAdmissionTimeoutError(
                    "prefetch queue empty and RAM headroom admission deadline expired"
                )
            time.sleep(self._wait_seconds)
        self.waited_for_ram_seconds += time.monotonic() - started
        return True

    def _fill(self, *, block_if_empty: bool) -> None:
        while (
            self._next_submit < len(self._rows)
            and len(self._futures) < self._max_ahead
        ):
            if not self._wait_for_headroom(block=block_if_empty and not self._futures):
                return
            index = self._next_submit
            self._futures[index] = self._executor.submit(_worker_prepare, self._rows[index])
            self._next_submit += 1

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self._next_consume >= len(self._rows):
            raise ValueError("prefetch consumer requested more rows than declared")
        expected = self._rows[self._next_consume]
        if row.get("query_id") != expected.get("query_id"):
            raise ValueError("prefetch consumption order differs from frozen manifest order")
        if self._next_consume not in self._futures:
            self._fill(block_if_empty=True)
        wait_started = time.perf_counter()
        prepared, memory = self._futures.pop(self._next_consume).result()
        consumer_wait_seconds = time.perf_counter() - wait_started
        self.worker_memory.append(cast(dict[str, int], memory))
        record = {
            "query_id": expected.get("query_id"),
            "worker_pid": int(memory["pid"]),
            "worker_prepare_seconds": int(memory["prepare_ns"]) / 1e9,
            "worker_rss_bytes": int(memory["rss_bytes"]),
            "worker_peak_rss_bytes": int(memory["peak_rss_bytes"]),
            "consumer_wait_seconds": consumer_wait_seconds,
            "consumer_prepare_receipt_semantics": (
                "frozen seconds.prepare measures consumer wait plus retrieval, not worker CPU total"
            ),
        }
        if len(self.execution_records) < self._hash_limit:
            record["prepared_sha256"] = prepared_sha256(prepared)
        self.execution_records.append(record)
        self._next_consume += 1
        # Never hold a completed result behind RAM admission.  Refill is best-effort
        # while another future exists; an empty queue waits only on the next call.
        self._fill(block_if_empty=False)
        return cast(dict[str, Any], prepared)

    def close(self) -> None:
        """Stop accepting work and terminate the bounded pool cleanly."""
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> OrderedPreparePrefetch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def pending_rows(output: Path, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return only rows without an intact saved fragment; never open model inputs."""
    pending: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        path = output / "predictions" / f"query_{index:05d}.json"
        checksum = path.with_suffix(".sha256")
        intact = (
            path.is_file()
            and checksum.is_file()
            and checksum.read_text(encoding="ascii").strip() == digest(path)
        )
        if not intact:
            pending.append(row)
    return pending


def backend_binding(
    manifest_path: Path, *, workers: int, max_ahead: int, min_available_bytes: int
) -> dict[str, Any]:
    """Return execution-only lineage, separate from scientific inference identity."""
    return {
        "artifact_type": "ordered_spawn_prefetch_backend_v1",
        "source_sha256": digest(Path(__file__)),
        "manifest_sha256": digest(manifest_path),
        "workers": workers,
        "max_ahead": max_ahead,
        "min_available_bytes": min_available_bytes,
        "worker_torch_threads": 2,
        "consumer_order": "frozen_manifest_order",
        "scientific_binding_changed": False,
        "optimizer_updates": 0,
    }


def run_prefetched(
    campaign: Path,
    output: Path,
    device: str,
    limit: int | None,
    *,
    workers: int = 2,
    max_ahead: int = 2,
) -> None:
    """Run the frozen evaluator with only its local preparation callable replaced."""
    from operational.evttc_transfer import inputs
    from operational.evttc_transfer import run as frozen_run

    manifest_path = output / "QUERY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    pending = pending_rows(output, rows)
    binding = backend_binding(
        manifest_path, workers=workers, max_ahead=max_ahead, min_available_bytes=2 * GIB
    )
    backend_path = output / "BACKEND_FREEZE.json"
    if backend_path.exists():
        if json.loads(backend_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("prefetch backend freeze differs from the existing execution lineage")
    else:
        atomic_json(backend_path, binding)
    original = inputs.prepare
    execution_path = output / "BACKEND_EXECUTION.json"
    previous_execution_sha256 = digest(execution_path) if execution_path.exists() else None
    execution_started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    try:
        prefetch = OrderedPreparePrefetch(
            pending,
            workers=workers,
            max_ahead=max_ahead,
            stop_requested=lambda: (output / "STOP_REQUEST").exists(),
        )
    except (PrefetchStoppedError, PrefetchAdmissionTimeoutError) as error:
        status = (
            "PAUSED_PRESERVED"
            if isinstance(error, PrefetchStoppedError)
            else "WAITING_FOR_RAM_TIMEOUT_PRESERVED"
        )
        receipt = {
            "artifact_type": "ordered_spawn_prefetch_execution_v1",
            "status": status,
            "started_utc": started_utc,
            "finished_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - execution_started,
            "error": str(error),
            "records": [],
            "previous_execution_sha256": previous_execution_sha256,
            "scientific_binding_changed": False,
        }
        atomic_json(execution_path, receipt)
        atomic_json(output / "STATE.json", receipt)
        return
    try:
        with prefetch:
            inputs.prepare = prefetch
            try:
                frozen_run.run(campaign, output, device, limit)
            finally:
                inputs.prepare = original
    finally:
        atomic_json(
            execution_path,
            {
                "artifact_type": "ordered_spawn_prefetch_execution_v1",
                "status": "FINISHED_OR_PRESERVED_BY_FROZEN_RUNNER",
                "started_utc": started_utc,
                "finished_utc": datetime.now(UTC).isoformat(),
                "wall_seconds": time.perf_counter() - execution_started,
                "records": prefetch.execution_records,
                "previous_execution_sha256": previous_execution_sha256,
                "queries_consumed": len(prefetch.execution_records),
                "waited_for_ram_seconds": prefetch.waited_for_ram_seconds,
                "scientific_binding_changed": False,
                "timing_warning": (
                    "Frozen fragment seconds.prepare is consumer wait/retrieval under prefetch; "
                    "use worker_prepare_seconds and backend wall_seconds for execution analysis, "
                    "and never sum overlapped per-query durations as end-to-end latency."
                ),
            },
        )


def _select_pilot_rows(
    all_rows: Sequence[Mapping[str, Any]], count: int, selection: str
) -> list[Mapping[str, Any]]:
    if selection == "sequential":
        return list(all_rows[:count])
    if selection != "first-per-sequence":
        raise ValueError(f"unknown pilot selection: {selection}")
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in all_rows:
        sequence = str(row["sequence_id"])
        if sequence not in seen:
            seen.add(sequence)
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def pilot(
    manifest_path: Path, output: Path, count: int = 4, selection: str = "sequential"
) -> dict[str, Any]:
    """Measure four real rows without running a model or reading TTC labels."""
    import psutil

    from operational.evttc_transfer.inputs import prepare

    all_rows = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"]
    rows = _select_pilot_rows(all_rows, count, selection)
    if len(rows) != count:
        raise ValueError(f"pilot requires {count} manifest rows")
    parent = psutil.Process()
    begin = time.perf_counter()
    with OrderedPreparePrefetch(rows, workers=2, max_ahead=2) as prefetch:
        process_outputs = []
        process_completion_seconds = []
        for row in rows:
            process_outputs.append(prefetch(row))
            process_completion_seconds.append(time.perf_counter() - begin)
        worker_memory = list(prefetch.worker_memory)
        waited = prefetch.waited_for_ram_seconds
    process_seconds = time.perf_counter() - begin
    process_parent_rss = int(parent.memory_info().rss)

    begin = time.perf_counter()
    direct_outputs = []
    direct_query_seconds = []
    for row in rows:
        query_begin = time.perf_counter()
        direct_outputs.append(prepare(row))
        direct_query_seconds.append(time.perf_counter() - query_begin)
    direct_seconds = time.perf_counter() - begin
    direct_parent_rss = int(parent.memory_info().rss)
    process_after_first = process_completion_seconds[-1] - process_completion_seconds[0]
    direct_after_first = sum(direct_query_seconds[1:])
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
    result = {
        "artifact_type": "evttc_prepare_prefetch_pilot_v1",
        "status": "PASSED",
        "queries": count,
        "selection": selection,
        "manifest_sha256": digest(manifest_path),
        "source_sha256": digest(Path(__file__)),
        "workers": 2,
        "max_ahead": 2,
        "process_seconds_including_spawn": process_seconds,
        "process_completion_seconds_including_spawn": process_completion_seconds,
        "process_seconds_after_first_completion": process_after_first,
        "direct_seconds_after_process_run": direct_seconds,
        "direct_query_seconds": direct_query_seconds,
        "steady_after_first_speedup": direct_after_first / process_after_first,
        "throughput_ratio_process_over_direct": direct_seconds / process_seconds,
        "parent_rss_bytes_after_process": process_parent_rss,
        "parent_rss_bytes_after_direct": direct_parent_rss,
        "worker_memory": worker_memory,
        "max_worker_peak_rss_bytes": max(item["peak_rss_bytes"] for item in worker_memory),
        "waited_for_ram_seconds": waited,
        "array_payload_bytes_per_query": [
            sum(value.nbytes for value in output.values() if isinstance(value, np.ndarray))
            for output in process_outputs
        ],
        "records": records,
        "labels_read": False,
        "models_run": False,
        "optimizer_updates": 0,
        "decision": (
            "QUALIFIED_ENABLE_ON_NEXT_RESUME"
            if direct_after_first > process_after_first
            else "DO_NOT_ENABLE_PROCESS_OVERHEAD_DOMINATES"
        ),
        "interpretation": (
            "Process timing includes spawn/import cost; direct timing runs second and may benefit "
            "from OS cache, so this is conservative for process prefetch. Deployment benefit also "
            "depends on overlap with GPU inference and is not inferred from this CPU-only ratio."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "PILOT.json", result)
    freeze = backend_binding(
        manifest_path, workers=2, max_ahead=2, min_available_bytes=2 * GIB
    )
    atomic_json(output / "BACKEND_FREEZE.json", freeze)
    atomic_json(
        output / "SHA256.json",
        {name: digest(output / name) for name in ("PILOT.json", "BACKEND_FREEZE.json")},
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--manifest", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    pilot_parser.add_argument("--count", type=int, default=4)
    pilot_parser.add_argument(
        "--selection", choices=("sequential", "first-per-sequence"), default="sequential"
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    run_parser.add_argument("--max-ahead", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.command == "pilot":
        result = pilot(
            args.manifest.resolve(), args.output.resolve(), args.count, args.selection
        )
        print(json.dumps(result, indent=2))
    else:
        run_prefetched(
            args.campaign.resolve(),
            args.output.resolve(),
            args.device,
            args.limit,
            workers=args.workers,
            max_ahead=args.max_ahead,
        )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
