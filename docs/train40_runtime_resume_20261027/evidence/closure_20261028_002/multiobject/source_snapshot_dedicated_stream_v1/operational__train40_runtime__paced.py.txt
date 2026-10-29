"""Independent, bounded host arrivals for honest persistent-service measurements.

The producer releases raw observations, never representations or predictions.
The caller owns GPU execution on its calling thread and must synchronize final
host outputs before returning. This runner measures a replay/capacity experiment;
it does not certify that a sensor or twenty real tracks produced the payloads.
"""

from __future__ import annotations

import math
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

Payload = TypeVar("Payload")


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
    first_ns = time.perf_counter_ns() + round(startup_lead_s * 1e9)
    pending: queue.Queue[Arrival[Payload]] = queue.Queue(maxsize=queue_capacity)
    stop, done = threading.Event(), threading.Event()
    receipts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    expected = {f"update-{index}": list(ids) for index in range(updates)}

    def produce() -> None:
        try:
            for index in range(updates):
                deadline = first_ns + round(index * 1e9 / rate_hz)
                delay_s = (deadline - time.perf_counter_ns()) / 1e9
                if stop.wait(max(0.0, delay_s)):
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
        finally:
            done.set()

    producer = threading.Thread(target=produce, name="train40-raw-arrivals", daemon=True)
    producer.start()
    try:
        while not done.is_set() or not pending.empty():
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
    finally:
        stop.set()
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
    }
    return {
        "schema": "train40_paced_raw_replay_v1",
        "scope": "HOST_PACED_RAW_REPLAY_CAPACITY_NOT_PHYSICAL_SENSOR_CERTIFICATION",
        "physical_goal_demonstrated": False,
        "capacity_checks_passed": all(capacity_checks.values()),
        "capacity_checks": capacity_checks,
        "rate_hz": rate_hz,
        "offered_updates": updates,
        "offered_duration_s": updates / rate_hz,
        "actual_wall_duration_s": (finish_ns - first_ns) / 1e9,
        "queue_capacity": queue_capacity,
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
        "timing_note": "All-object host completion; no division by number of objects",
    }
