"""Independent, bounded host arrivals for honest persistent-service measurements.

The producer releases raw observations, never representations or predictions.
The caller owns GPU execution on its calling thread and must synchronize final
host outputs before returning. This runner measures a replay/capacity experiment;
it does not certify that a sensor or twenty real tracks produced the payloads.
"""

from __future__ import annotations

import math
import platform
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from operational.train40_runtime.measurement import (
    PREDICTION_COMPONENTS,
    group_updates,
    latency_distribution,
)
from operational.train40_runtime.paced_statistics import paced_latency_populations

Payload = TypeVar("Payload")
SLEEP_STOP_POLL_SECONDS = 0.010


def pacing_clock_metadata(wait_strategy: str) -> dict[str, Any]:
    """Describe the actual Python/clock path without claiming observed precision."""
    if wait_strategy not in ("event", "sleep"):
        raise ValueError("wait_strategy must be 'event' or 'sleep'")
    clock = time.get_clock_info("perf_counter")
    return {
        "wait_strategy": wait_strategy,
        "wait_primitive": "threading.Event.wait" if wait_strategy == "event" else "time.sleep",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "deadline_clock": "time.perf_counter_ns",
        "clock_implementation": clock.implementation,
        "clock_monotonic": clock.monotonic,
        "clock_adjustable": clock.adjustable,
        "clock_resolution_seconds": clock.resolution,
        "sleep_stop_poll_seconds": SLEEP_STOP_POLL_SECONDS if wait_strategy == "sleep" else None,
        "deadlines_absolute": True,
        "busy_spin": False,
        "scheduler_lateness_subtracted": False,
    }


def wait_until_deadline(
    deadline_ns: int,
    stop: threading.Event,
    *,
    wait_strategy: str,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait for an absolute deadline; return True if cancellation was requested.

    The legacy Event path is unchanged. The opt-in sleep path uses Python's
    time.sleep (a high-resolution waitable timer on Windows with Python 3.11+),
    recalculating the remaining time after every wake. Sleep requests are at
    most 10ms so cancellation is polled without a busy spin. OS scheduling may
    delay any wake, so this is not a guaranteed wall-clock cancellation bound.
    """
    if wait_strategy == "event":
        return stop.wait(max(0.0, (deadline_ns - clock_ns()) / 1e9))
    if wait_strategy != "sleep":
        raise ValueError("wait_strategy must be 'event' or 'sleep'")
    while not stop.is_set():
        remaining_ns = deadline_ns - clock_ns()
        if remaining_ns <= 0:
            return False
        sleep(min(remaining_ns / 1e9, SLEEP_STOP_POLL_SECONDS))
    return True


@dataclass(frozen=True)
class Arrival(Generic[Payload]):
    """Raw payload released on an independent fixed monotonic schedule."""

    update_id: str
    scheduled_ns: int
    observation_available_ns: int
    payload: Payload


def _finite_predictions(result: Mapping[str, Any]) -> bool:
    predictions = result.get("predictions")
    if not isinstance(predictions, Mapping):
        return False
    for component in PREDICTION_COMPONENTS:
        if component not in predictions:
            return False
        value = predictions[component]
        if component == "R_CTX_TRAIN40" and value is None and result.get("rgb_available") is False:
            continue
        if isinstance(value, (bool, str, bytes)) or value is None:
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def run_paced_replay(
    acquire_raw: Callable[[int], Payload],
    infer_all: Callable[[Arrival[Payload]], Mapping[str, Mapping[str, Any]]],
    object_ids: Sequence[str],
    *,
    updates: int,
    rate_hz: float = 50.0,
    queue_capacity: int = 2,
    startup_lead_s: float = 0.05,
    wait_strategy: str = "event",
    source_warmup: Callable[[], Mapping[str, Any] | None] | None = None,
    source_finalize: Callable[[], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Measure all scheduled arrivals, retaining failures and overload drops.

    ``acquire_raw(index)`` runs on the producer thread at the fixed deadline.
    It may read/release raw observations and metadata; it must not crop ROIs,
    encode events, run models, or touch CUDA. Each returned payload must own its
    bytes until consumed (no mutable shared scratch arrays). Loading delay stays
    visible in both the arrival receipt and deadline-to-output response.

    ``infer_all`` runs serially on the caller thread and returns one mapping per
    required object, including ``success: bool`` and all three TTC predictions
    under ``predictions`` using the frozen component names. Only RGB may be null,
    with explicit ``rgb_available=False``. Nonfinite/missing outputs fail. Its
    completion clock follows the last available host TTC. Queue overflow drops
    the new update with an explicit receipt. It does not postpone future arrivals
    or silently replace pending observations. Exceptions become failed receipts.
    An incomplete/drop/source-error run cannot pass the capacity checks.
    ``wait_strategy='event'`` preserves the v1 scheduler. Explicit ``'sleep'``
    uses bounded time.sleep waits against the same absolute deadlines. All
    scheduler lateness remains inside both acquisition and response receipts.
    Optional source lifecycle callbacks execute on the producer thread. A
    successful warmup sets the first deadline once, after its declared startup
    work; later deadlines remain fixed. Finalization always runs in that same
    thread, after the consumer drains offered work or aborts. The producer marks
    offering complete before waiting for finalization permission, so its cleanup
    cannot compete with the last inference. This also applies after warmup or
    acquisition failure. Caller cancellation releases the producer in finally.
    """
    ids = tuple(object_ids)
    if not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("object_ids must be nonempty unique strings")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    if (
        isinstance(queue_capacity, bool)
        or not isinstance(queue_capacity, int)
        or queue_capacity <= 0
    ):
        raise ValueError("queue_capacity must be a positive integer")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    if not math.isfinite(startup_lead_s) or startup_lead_s < 0:
        raise ValueError("startup_lead_s must be finite and nonnegative")
    scheduler = pacing_clock_metadata(wait_strategy)
    first_ns = time.perf_counter_ns() + round(startup_lead_s * 1e9)
    pending: queue.Queue[Arrival[Payload]] = queue.Queue(maxsize=queue_capacity)
    stop, offering_done, permit_finalize = (threading.Event(), threading.Event(), threading.Event())
    receipts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    expected = {f"update-{index}": list(ids) for index in range(updates)}
    lifecycle: dict[str, Any] = {
        "schema": "train40_paced_source_lifecycle_v2",
        "order": "OFFERING_DONE_THEN_CONSUMER_DRAIN_OR_ABORT_THEN_FINALIZE",
        "warmup": None,
        "finalize": None,
        "errors": [],
        "offering_ended_ns": None,
        "consumer_drain_completed_ns": None,
        "consumer_stopped_ns": None,
        "consumer_queue_drained": False,
        "finalize_permitted_ns": None,
        "producer_finished_ns": None,
    }

    def source_callback(name: str, callback: Callable[[], Mapping[str, Any] | None]) -> None:
        receipt: dict[str, Any] = {
            "started_ns": time.perf_counter_ns(),
            "thread_id": threading.get_ident(),
        }
        lifecycle[name] = receipt
        try:
            result = callback()
            if result is not None and not isinstance(result, Mapping):
                raise TypeError("Source lifecycle callbacks must return a mapping or None")
            receipt["result"] = dict(result) if result is not None else None
        finally:
            receipt["ended_ns"] = time.perf_counter_ns()

    def produce() -> None:
        nonlocal first_ns
        phase = "warmup"
        try:
            if source_warmup is not None:
                source_callback("warmup", source_warmup)
                first_ns = time.perf_counter_ns() + round(startup_lead_s * 1e9)
            phase = "acquisition"
            for index in range(updates):
                deadline = first_ns + round(index * 1e9 / rate_hz)
                if wait_until_deadline(deadline, stop, wait_strategy=wait_strategy):
                    break
                begin = time.perf_counter_ns()
                receipt: dict[str, Any] = {
                    "update_id": f"update-{index}",
                    "scheduled_ns": deadline,
                    "acquisition_started_ns": begin,
                }
                try:
                    payload = acquire_raw(index)
                except Exception as exc:
                    receipt.update(
                        status="SOURCE_ERROR",
                        error=f"{type(exc).__name__}: {exc}",
                        acquisition_ended_ns=time.perf_counter_ns(),
                    )
                    receipts.append(receipt)
                    continue
                arrived = time.perf_counter_ns()
                receipt["observation_available_ns"] = arrived
                arrival = Arrival(receipt["update_id"], deadline, arrived, payload)
                try:
                    pending.put_nowait(arrival)
                    receipt["status"] = "ENQUEUED"
                except queue.Full:
                    receipt["status"] = "DROPPED_QUEUE_FULL"
                receipts.append(receipt)
        except BaseException as exc:
            lifecycle["errors"].append({"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            lifecycle["offering_ended_ns"] = time.perf_counter_ns()
            offering_done.set()
            # Keep source ownership and local payloads alive until the consumer
            # finishes. offering_done, not finalization, terminates its loop.
            permit_finalize.wait()
            if source_finalize is not None:
                try:
                    source_callback("finalize", source_finalize)
                except BaseException as exc:
                    lifecycle["errors"].append(
                        {"phase": "finalize", "error": f"{type(exc).__name__}: {exc}"}
                    )
            lifecycle["producer_finished_ns"] = time.perf_counter_ns()

    producer = threading.Thread(target=produce, name="train40-raw-arrivals", daemon=True)
    producer.start()
    consumer_drained = False
    try:
        while not offering_done.is_set() or not pending.empty():
            try:
                arrival = pending.get(timeout=0.01)
            except queue.Empty:
                continue
            started = time.perf_counter_ns()
            error = None
            result: Mapping[str, Mapping[str, Any]] = {}
            try:
                result = infer_all(arrival)
                if not isinstance(result, Mapping) or any(
                    not isinstance(key, str) or not isinstance(value, Mapping)
                    for key, value in result.items()
                ):
                    raise TypeError("infer_all must return an object-ID to result mapping")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = {}
            completed = time.perf_counter_ns()
            # Unexpected IDs remain explicit records; missing IDs fail coverage.
            for identity in dict.fromkeys((*ids, *result)):
                item = dict(result.get(identity, {}))
                records.append(
                    {
                        "update_id": arrival.update_id,
                        "object_id": identity,
                        "scheduled_ns": arrival.scheduled_ns,
                        "observation_available_ns": arrival.observation_available_ns,
                        "started_ns": started,
                        "completed_ns": completed,
                        "success": error is None
                        and item.get("success") is True
                        and _finite_predictions(item),
                        "result": item,
                        "error": error or (None if identity in result else "MISSING_OBJECT"),
                    }
                )
            pending.task_done()
        consumer_drained = True
    finally:
        stop.set()
        stopped_ns = time.perf_counter_ns()
        lifecycle["consumer_stopped_ns"] = stopped_ns
        lifecycle["consumer_queue_drained"] = consumer_drained
        lifecycle["consumer_drain_completed_ns"] = stopped_ns if consumer_drained else None
        lifecycle["finalize_permitted_ns"] = time.perf_counter_ns()
        permit_finalize.set()
        producer.join()
    finish_ns = time.perf_counter_ns()
    grouped = group_updates(records, expected)
    by_update = {row["update_id"]: row for row in grouped["updates"]}
    available = [row for row in receipts if "observation_available_ns" in row]
    processed = [row for row in available if "completed_ns" in by_update[row["update_id"]]]
    distribution = lambda xs: latency_distribution(xs) if xs else None  # noqa: E731
    received_ids = [row["update_id"] for row in receipts]
    complete_arrivals = received_ids == list(expected)
    no_drops = all(row["status"] == "ENQUEUED" for row in receipts)
    responses = [
        (by_update[row["update_id"]]["completed_ns"] - row["scheduled_ns"]) / 1e6
        for row in processed
    ]
    arrival_responses = [
        (by_update[row["update_id"]]["completed_ns"] - row["observation_available_ns"]) / 1e6
        for row in processed
    ]
    end_times = [row["completed_ns"] for row in grouped["updates"] if "completed_ns" in row]
    successful_end_times = [
        row["completed_ns"] for row in grouped["updates"] if row["passed"] is True
    ]
    pending_at_deadlines = [
        sum(
            prior["scheduled_ns"] < row["scheduled_ns"] < prior["completed_ns"]
            for prior in grouped["updates"]
            if "completed_ns" in prior
        )
        for row in receipts
    ]
    capacity_checks = {
        "all_scheduled_arrivals_recorded": complete_arrivals,
        "no_drops_or_source_errors": no_drops and complete_arrivals,
        "complete_object_coverage": grouped["passed"],
        "actual_availability_before_service": bool(records)
        and all(
            row["scheduled_ns"]
            <= row["observation_available_ns"]
            <= row["started_ns"]
            <= row["completed_ns"]
            for row in records
        ),
        "every_deadline_response_under_20ms": bool(responses) and max(responses) < 20.0,
        "no_prior_work_at_scheduled_arrivals": bool(pending_at_deadlines)
        and max(pending_at_deadlines) == 0,
        "finished_before_end_of_last_period": bool(end_times)
        and max(end_times) < first_ns + round(updates * 1e9 / rate_hz),
        "offered_50hz_for_at_least_60s": rate_hz == 50.0 and updates / rate_hz >= 60.0,
        "twenty_distinct_objects_per_update": len(ids) == 20,
        "source_lifecycle_succeeded": not lifecycle["errors"],
    }
    summary: dict[str, Any] = {
        "schema": "train40_paced_raw_replay_v1",
        "scope": "HOST_PACED_RAW_REPLAY_CAPACITY_NOT_PHYSICAL_SENSOR_CERTIFICATION",
        "physical_goal_demonstrated": False,
        "capacity_checks_passed": all(capacity_checks.values()),
        "capacity_checks": capacity_checks,
        "rate_hz": rate_hz,
        "offered_updates": updates,
        "offered_duration_s": updates / rate_hz,
        "actual_wall_duration_s": (finish_ns - first_ns) / 1e9,
        "actual_wall_duration_includes_source_finalize_and_join": True,
        "last_response_duration_s": (max(end_times) - first_ns) / 1e9 if end_times else None,
        "last_successful_all_object_ttc_duration_s": (
            (max(successful_end_times) - first_ns) / 1e9 if successful_end_times else None
        ),
        "queue_capacity": queue_capacity,
        "wait_strategy": wait_strategy,
        "scheduler": scheduler,
        "first_scheduled_ns": first_ns,
        "source_lifecycle": lifecycle,
        "object_ids": list(ids),
        "arrival_receipts": receipts,
        "object_records": records,
        "grouped": grouped,
        "missing_arrival_ids": sorted(set(expected) - set(received_ids)),
        "dropped_updates": sum(row["status"] == "DROPPED_QUEUE_FULL" for row in receipts),
        "source_errors": sum(row["status"] == "SOURCE_ERROR" for row in receipts),
        "deadline_response_ms": distribution(responses),
        "raw_availability_response_ms": distribution(arrival_responses),
        "raw_availability_lateness_ms": distribution(
            [(row["observation_available_ns"] - row["scheduled_ns"]) / 1e6 for row in available]
        ),
        "prior_work_at_each_scheduled_arrival": pending_at_deadlines,
        "legacy_response_population": "ALL_RESPONSES_INCLUDING_FAILURES",
        "timing_note": "Legacy responses include failed acknowledgements; valid all-object "
        "TTC distributions are separate in latency_populations. No division by object count.",
    }
    summary["latency_populations"] = paced_latency_populations(summary)
    return summary
