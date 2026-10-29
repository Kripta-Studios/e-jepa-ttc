"""Conservative parity and timing gates for the immutable TRAIN40 candidate.

These functions analyze evidence; they do not manufacture timings, infer missing
objects, relax tolerances, read TTC labels, or write the frozen candidate.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PREDICTION_COMPONENTS = ("PHASE_ONLY_TRAIN40_S7", "E_CTX_TRAIN40", "R_CTX_TRAIN40")
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-5
Record = Mapping[str, Any]


def coverage_report(
    expected_ids: Sequence[str], records: Sequence[Record], *, id_key: str = "query_id"
) -> dict[str, Any]:
    """Count every expected physical ID, retaining duplicates and unexpected IDs."""
    if not expected_ids or any(not isinstance(value, str) or not value for value in expected_ids):
        raise ValueError("Expected IDs must be nonempty strings in a nonempty population")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected population contains duplicate physical IDs")
    ids = [row.get(id_key) for row in records]
    invalid = [index for index, value in enumerate(ids) if not isinstance(value, str) or not value]
    counts = Counter(value for value in ids if isinstance(value, str) and value)
    expected = set(expected_ids)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    missing, unexpected = sorted(expected - counts.keys()), sorted(counts.keys() - expected)
    return {
        "passed": not (invalid or duplicates or missing or unexpected),
        "expected": len(expected_ids),
        "records": len(records),
        "unique_present": len(expected & counts.keys()),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "duplicate_ids": duplicates,
        "invalid_id_record_indices": invalid,
    }


def _numeric_state(value: object) -> tuple[str, float | None]:
    if value is None:
        return "null", None
    if isinstance(value, (bool, str, bytes)):
        return "invalid", None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return "invalid", None
    if math.isnan(number):
        return "nan", None
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf", None
    return "finite", number


def prediction_parity(
    reference: Sequence[Record], candidate: Sequence[Record], expected_ids: Sequence[str]
) -> dict[str, Any]:
    """Compare all three TTC outputs and causal-history metadata at fixed tolerances.

    A null RGB output is valid only with an explicit RGB-unavailable declaration.
    Missing fields, nonfinite final/event outputs, duplicate IDs and coverage loss
    fail even if the surviving finite outputs are numerically identical.
    """
    old_coverage = coverage_report(expected_ids, reference)
    new_coverage = coverage_report(expected_ids, candidate)
    old = {row["query_id"]: row for row in reference if isinstance(row.get("query_id"), str)}
    new = {row["query_id"]: row for row in candidate if isinstance(row.get("query_id"), str)}
    summaries: dict[str, dict[str, Any]] = {
        name: {"finite_pairs": 0, "max_abs_error": 0.0, "max_relative_error": 0.0}
        for name in PREDICTION_COMPONENTS
    }
    details = []
    metadata_keys = ("history_lengths", "rgb_available", "rgb_unavailable_reason")
    for query_id in expected_ids:
        if query_id not in old or query_id not in new:
            details.append({"query_id": query_id, "passed": False, "reason": "missing_record"})
            continue
        baseline, proposed = old[query_id], new[query_id]
        metadata_errors = [
            key
            for key in metadata_keys
            if key not in baseline or key not in proposed or baseline[key] != proposed[key]
        ]
        components = {}
        for name in PREDICTION_COMPONENTS:
            before = baseline.get("predictions", {})
            after = proposed.get("predictions", {})
            present = (
                isinstance(before, Mapping)
                and isinstance(after, Mapping)
                and name in before
                and name in after
            )
            ref_state, ref = _numeric_state(before[name]) if present else ("missing", None)
            new_state, val = _numeric_state(after[name]) if present else ("missing", None)
            finite_pair = ref_state == new_state == "finite"
            absent_rgb = (
                name == "R_CTX_TRAIN40"
                and ref_state == new_state == "null"
                and baseline.get("rgb_available") is False
                and proposed.get("rgb_available") is False
            )
            delta = abs(val - ref) if finite_pair and val is not None and ref is not None else None
            limit = PARITY_ATOL + PARITY_RTOL * abs(ref) if ref is not None else None
            passed = absent_rgb or (delta is not None and limit is not None and delta <= limit)
            exact = absent_rgb or (finite_pair and val == ref)
            components[name] = {
                "passed": passed,
                "exact": exact,
                "reference_state": ref_state,
                "candidate_state": new_state,
                "finite_mask_equal": (ref_state == "finite") == (new_state == "finite"),
                "abs_error": delta,
                "allowed_abs_error": limit,
            }
            if delta is not None and ref is not None:
                summary = summaries[name]
                summary["finite_pairs"] += 1
                summary["max_abs_error"] = max(summary["max_abs_error"], delta)
                summary["max_relative_error"] = max(
                    summary["max_relative_error"], delta / max(abs(ref), PARITY_ATOL)
                )
        details.append(
            {
                "query_id": query_id,
                "passed": not metadata_errors
                and all(item["passed"] for item in components.values()),
                "exact": not metadata_errors and all(item["exact"] for item in components.values()),
                "metadata_mismatches": metadata_errors,
                "components": components,
            }
        )
    coverage_ok = old_coverage["passed"] and new_coverage["passed"]
    return {
        "passed": coverage_ok and all(row["passed"] for row in details),
        "exact": coverage_ok and all(row.get("exact", False) for row in details),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "reference_coverage": old_coverage,
        "candidate_coverage": new_coverage,
        "components": summaries,
        "failed_query_ids": [row["query_id"] for row in details if not row["passed"]],
        "per_query": details,
    }


def latency_distribution(milliseconds: Sequence[float]) -> dict[str, Any]:
    """Summarize finite nonnegative wall durations, without inferring sustained Hz."""
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latency samples must be a nonempty vector of finite nonnegative ms")
    mean = float(values.mean())
    p50, p95, p99 = (float(value) for value in np.percentile(values, [50, 95, 99], method="linear"))
    return {
        "count": len(values),
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "quantile_method": "linear",
        "strictly_under_20ms_count": int((values < 20.0).sum()),
        "at_least_20ms_count": int((values >= 20.0).sum()),
        "over_50ms_count": int((values > 50.0).sum()),
        "over_100ms_count": int((values > 100.0).sum()),
        "inverse_mean_capacity_hz_not_sustained_throughput": 1000.0 / mean if mean else None,
    }


def _timestamp(row: Record, key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer from the same monotonic clock")
    return value


def group_updates(
    records: Sequence[Record], expected_objects: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Group object completions into all-object update latency and coverage.

    Each record has update_id, object_id, scheduled_ns, started_ns, completed_ns,
    and success. scheduled_ns is the common release time for the complete update;
    time spent waiting in the queue therefore remains inside response_ms.
    """
    if not expected_objects or any(not isinstance(key, str) or not key for key in expected_objects):
        raise ValueError("A nonempty explicit update/object population is required")
    expected_update_ids = list(expected_objects)
    unexpected = sorted(
        {
            str(row.get("update_id"))
            for row in records
            if row.get("update_id") not in expected_objects
        }
    )
    by_update: dict[str, list[Record]] = {key: [] for key in expected_objects}
    for row in records:
        identity = row.get("update_id")
        if isinstance(identity, str) and identity in by_update:
            by_update[identity].append(row)
    updates = []
    for update_id in expected_update_ids:
        members = by_update[update_id]
        coverage = coverage_report(expected_objects[update_id], members, id_key="object_id")
        errors: list[str] = []
        if not members:
            errors.append("no_object_results")
            updates.append(
                {
                    "update_id": update_id,
                    "passed": False,
                    "coverage": coverage,
                    "expected_object_ids": list(expected_objects[update_id]),
                    "completed_object_ids": [],
                }
            )
            continue
        schedule = [_timestamp(row, "scheduled_ns") for row in members]
        starts = [_timestamp(row, "started_ns") for row in members]
        ends = [_timestamp(row, "completed_ns") for row in members]
        if len(set(schedule)) != 1:
            errors.append("object_release_times_disagree")
        if any(
            start < release or end < start
            for release, start, end in zip(schedule, starts, ends, strict=True)
        ):
            errors.append("invalid_timestamp_order")
        if not all(row.get("success") is True for row in members):
            errors.append("unsuccessful_object")
        release, start, end = min(schedule), min(starts), max(ends)
        updates.append(
            {
                "update_id": update_id,
                "passed": coverage["passed"] and not errors,
                "coverage": coverage,
                "errors": errors,
                "objects": len(expected_objects[update_id]),
                "expected_object_ids": list(expected_objects[update_id]),
                "completed_object_ids": [
                    row["object_id"]
                    for row in members
                    if row.get("success") is True
                    and isinstance(row.get("object_id"), str)
                    and row["object_id"]
                ],
                "scheduled_ns": release,
                "started_ns": start,
                "completed_ns": end,
                "response_ms": (end - release) / 1e6,
                "service_span_ms": (end - start) / 1e6,
                "queue_wait_ms": (start - release) / 1e6,
            }
        )
    return {
        "passed": not unexpected and all(update["passed"] for update in updates),
        "unexpected_update_ids": unexpected,
        "expected_updates": len(expected_objects),
        "expected_object_results": sum(len(ids) for ids in expected_objects.values()),
        "updates": updates,
    }


def _object_capacity(grouped: Record, required: int) -> dict[str, Any]:
    """Require actual distinct successful object identities, not scope flags alone."""
    details = []
    for row in grouped.get("updates", []):
        expected, completed = row.get("expected_object_ids"), row.get("completed_object_ids")
        valid_ids = all(
            isinstance(ids, (list, tuple))
            and all(isinstance(identity, str) and identity for identity in ids)
            for ids in (expected, completed)
        )
        expected = list(expected) if valid_ids else []
        completed = list(completed) if valid_ids else []
        counts = Counter(completed)
        duplicates = sorted(identity for identity, count in counts.items() if count > 1)
        missing = sorted(set(expected) - set(completed))
        unexpected = sorted(set(completed) - set(expected))
        passed = (
            valid_ids
            and row.get("passed") is True
            and row.get("coverage", {}).get("passed") is True
            and len(expected) == len(set(expected)) == required
            and len(completed) == len(set(completed)) == required
            and not missing
            and not unexpected
        )
        details.append(
            {
                "update_id": row.get("update_id"),
                "passed": passed,
                "declared_objects": len(expected),
                "successful_object_records": len(completed),
                "distinct_successful_objects": len(counts),
                "missing_object_ids": missing,
                "unexpected_object_ids": unexpected,
                "duplicate_completed_object_ids": duplicates,
            }
        )
    return {
        "passed": bool(details)
        and len(details) == grouped.get("expected_updates")
        and all(row["passed"] for row in details),
        "required_objects_per_update": required,
        "expected_updates": grouped.get("expected_updates"),
        "updates_meeting_capacity": sum(row["passed"] for row in details),
        "per_update": details,
    }


def scheduled_stream_report(
    grouped: Record,
    scope: Record,
    *,
    rate_hz: float = 50.0,
    minimum_duration_s: float = 60.0,
    required_objects_per_update: int = 20,
) -> dict[str, Any]:
    """Gate measured paced all-object updates; never certify a primary-only replay.

    The strict goal requires every complete response below 20ms, not merely its
    mean. The looser requested p95<=50ms/p99<=100ms checks are also reported.
    Sixty seconds is the default prospective sustained-load observation period.
    The task goal requires exactly twenty distinct successful objects per update;
    smaller explicit capacities remain separately labeled engineering probes.
    Scope declarations must be substantiated by the calling runner's evidence.
    """
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    if not math.isfinite(minimum_duration_s) or minimum_duration_s <= 0:
        raise ValueError("minimum_duration_s must be finite and positive")
    if (
        not isinstance(required_objects_per_update, int)
        or isinstance(required_objects_per_update, bool)
        or required_objects_per_update <= 0
    ):
        raise ValueError("required_objects_per_update must be a positive integer")
    object_capacity = _object_capacity(grouped, required_objects_per_update)
    required_scope: dict[str, object] = {
        "timing_source": "measured_monotonic_wall",
        "input_boundary": "last_required_raw_observation_available",
        "output_boundary": "all_required_object_ttc_available",
        "includes_preprocessing": True,
        "includes_transfers": True,
        "includes_complete_model_chain": True,
        "causal_observations": True,
        "paced_physical_arrivals": True,
        "all_required_objects_declared": True,
        "primary_only_cohort": False,
    }
    scope_failures = [
        key
        for key, expected in required_scope.items()
        if (
            scope.get(key) is not expected
            if isinstance(expected, bool)
            else scope.get(key) != expected
        )
    ]
    updates = sorted(
        [row for row in grouped.get("updates", []) if "scheduled_ns" in row],
        key=lambda row: row["scheduled_ns"],
    )
    if not updates:
        return {
            "passed": False,
            "reason": "no_timed_updates",
            "scope_failures": scope_failures,
            "object_capacity": object_capacity,
        }
    release = [_timestamp(row, "scheduled_ns") for row in updates]
    start = [_timestamp(row, "started_ns") for row in updates]
    end = [_timestamp(row, "completed_ns") for row in updates]
    if any(a > b or b > c for a, b, c in zip(release, start, end, strict=True)):
        raise ValueError("Invalid update timestamp order")
    period_ns = 1e9 / rate_hz
    cadence_error = [
        abs((value - release[0]) - index * period_ns) for index, value in enumerate(release)
    ]
    cadence_ok = max(cadence_error) <= 1.0
    offered_duration_s = len(updates) / rate_hz
    actual_elapsed_s = (max(max(end), release[-1] + period_ns) - release[0]) / 1e9
    # Outstanding includes work already executing; queued excludes it.
    outstanding, queued = [], []
    unfinished: list[int] = []
    not_started: list[int] = []
    for arrival, first, last in zip(release, start, end, strict=True):
        while unfinished and unfinished[0] <= arrival:
            heapq.heappop(unfinished)
        while not_started and not_started[0] <= arrival:
            heapq.heappop(not_started)
        outstanding.append(len(unfinished))
        queued.append(len(not_started))
        heapq.heappush(unfinished, last)
        heapq.heappush(not_started, first)
    final_pending = sum(last > release[-1] + period_ns for last in end)
    no_backlog = max(outstanding) == 0 and final_pending == 0
    response = latency_distribution(
        [(last - arrival) / 1e6 for arrival, last in zip(release, end, strict=True)]
    )
    service = latency_distribution(
        [(last - first) / 1e6 for first, last in zip(start, end, strict=True)]
    )
    wait = latency_distribution(
        [(first - arrival) / 1e6 for arrival, first in zip(release, start, strict=True)]
    )
    gates = {
        "complete_object_coverage": grouped.get("passed") is True,
        "required_distinct_objects_per_update": object_capacity["passed"],
        "declared_physical_pipeline_scope": not scope_failures,
        "paced_50hz": cadence_ok and rate_hz == 50.0,
        "sustained_observation_duration": offered_duration_s >= minimum_duration_s,
        "every_complete_update_strictly_under_20ms": response["max_ms"] < 20.0,
        "p95_at_most_50ms": response["p95_ms"] <= 50.0,
        "p99_at_most_100ms": response["p99_ms"] <= 100.0,
        "no_work_outstanding_at_next_arrival": no_backlog,
    }
    successful_updates = sum(row.get("passed") is True for row in updates)
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "scope": dict(scope),
        "scope_failures": scope_failures,
        "required_objects_per_update": required_objects_per_update,
        "object_capacity": object_capacity,
        "offered_rate_hz": rate_hz,
        "timed_update_count": len(updates),
        "successful_complete_update_count": successful_updates,
        "observed_completed_updates_per_wall_second": successful_updates / actual_elapsed_s,
        "offered_duration_s": offered_duration_s,
        "actual_elapsed_s": actual_elapsed_s,
        "minimum_duration_s": minimum_duration_s,
        "cadence_max_error_ns": max(cadence_error),
        "response": response,
        "service_span": service,
        "queue_wait": wait,
        "max_outstanding_before_arrival": max(outstanding),
        "max_queued_before_arrival": max(queued),
        "pending_after_last_period": final_pending,
        "outstanding_before_each_arrival": outstanding,
        "queued_before_each_arrival": queued,
        "backlog_policy": "Conservative: zero prior unfinished work at every scheduled arrival",
        "latency_scope": "All objects, including queue wait; no division by object count",
    }


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Hash only the explicitly enumerated immutable files, without writing them."""
    result = {}
    for path in sorted({path.resolve(strict=True) for path in paths}):
        if not path.is_file():
            raise ValueError(f"Expected an immutable regular file: {path}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        result[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
    if not result:
        raise ValueError("Cannot verify an empty immutable inventory")
    return result


def verify_snapshot(before: Mapping[str, Record]) -> dict[str, Any]:
    """Detect modified or missing enumerated files after an optimization run."""
    if not before:
        raise ValueError("Cannot verify an empty immutable inventory")
    changed, missing = [], []
    for name, expected in before.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name)
        elif snapshot_files([path])[str(path.resolve())] != dict(expected):
            changed.append(name)
    return {
        "passed": not (changed or missing),
        "files": len(before),
        "changed": changed,
        "missing": missing,
    }
