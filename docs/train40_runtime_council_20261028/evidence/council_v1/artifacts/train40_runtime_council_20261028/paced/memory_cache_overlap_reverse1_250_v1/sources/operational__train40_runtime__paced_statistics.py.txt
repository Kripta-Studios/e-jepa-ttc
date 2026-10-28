"""Pure receipt analysis separating valid all-object TTC from failed acknowledgements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _distribution(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("Paced population latencies must be finite and nonnegative")

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "mean_ms": math.fsum(ordered) / len(ordered),
        "min_ms": ordered[0],
        "p50_ms": quantile(0.5),
        "p95_ms": quantile(0.95),
        "p99_ms": quantile(0.99),
        "max_ms": ordered[-1],
        "at_least_20ms_count": sum(value >= 20 for value in ordered),
    }


def paced_latency_populations(paced: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive populations from original clocks without changing any admission gate.

    Missing historical per-update evidence returns None, never guessed successes.
    The successful population requires grouped.passed and, when object records are
    present, successful records for every expected object. Empty populations have
    count zero and null distributions. Drops and missing updates have no latency.
    """
    grouped = paced.get("grouped")
    arrivals = paced.get("arrival_receipts")
    if not isinstance(grouped, Mapping) or not isinstance(arrivals, list):
        return None
    updates = grouped.get("updates")
    if not isinstance(updates, list):
        return None

    def indexed(rows: list[Any]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("update_id"), str):
                raise ValueError("Malformed paced population update receipt")
            identity = row["update_id"]
            if identity in result:
                raise ValueError("Duplicate paced population update receipt")
            result[identity] = row
        return result

    by_update, by_arrival = indexed(updates), indexed(arrivals)
    records = paced.get("object_records")
    by_object_update: dict[str, list[Mapping[str, Any]]] = {}
    if records is not None:
        if not isinstance(records, list):
            raise ValueError("Malformed paced object records")
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("update_id"), str):
                raise ValueError("Malformed paced object record")
            by_object_update.setdefault(record["update_id"], []).append(record)

    def clock(row: Mapping[str, Any], name: str) -> int:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid paced population clock: {name}")
        return value

    populations: dict[str, list[dict[str, float]]] = {
        "successful_all_object_ttc": [],
        "failed_acknowledgements": [],
    }
    for identity, row in by_update.items():
        if "completed_ns" not in row:
            continue
        arrival = by_arrival.get(identity)
        if arrival is None:
            raise ValueError("Paced completion has no arrival receipt")
        scheduled = clock(arrival, "scheduled_ns")
        available = clock(arrival, "observation_available_ns")
        started, completed = clock(row, "started_ns"), clock(row, "completed_ns")
        if not scheduled <= available <= started <= completed:
            raise ValueError("Paced population clocks are out of order")
        passed = row.get("passed") is True
        if arrival.get("status") != "ENQUEUED":
            raise ValueError("Paced completion contradicts its arrival status")
        if passed and (
            not isinstance(row.get("coverage"), Mapping)
            or row["coverage"].get("passed") is not True
        ):
            raise ValueError("Successful paced group has incomplete coverage")
        if passed and records is not None:
            members = by_object_update.get(identity, [])
            expected = row.get("expected_object_ids")
            observed = [member.get("object_id") for member in members]
            if (
                not isinstance(expected, list)
                or not expected
                or not all(isinstance(identity, str) for identity in [*expected, *observed])
                or len(set(expected)) != len(expected)
                or len(expected) != len(observed)
                or sorted(expected) != sorted(str(identity) for identity in observed)
                or not all(member.get("success") is True for member in members)
            ):
                raise ValueError("Successful paced group contradicts its object records")
            if (
                any(clock(member, "scheduled_ns") != scheduled for member in members)
                or any(clock(member, "observation_available_ns") != available for member in members)
                or min(clock(member, "started_ns") for member in members) != started
                or max(clock(member, "completed_ns") for member in members) != completed
            ):
                raise ValueError("Successful paced group contradicts its object clocks")
        population = "successful_all_object_ttc" if passed else "failed_acknowledgements"
        populations[population].append(
            {
                "deadline_response_ms": (completed - scheduled) / 1e6,
                "raw_availability_response_ms": (completed - available) / 1e6,
                "service_span_ms": (completed - started) / 1e6,
            }
        )
    all_responses = [row for rows in populations.values() for row in rows]
    populations["all_responses_including_failures"] = all_responses
    expected_count = grouped.get("expected_updates")
    offered_count = paced.get("offered_updates")
    counts = {
        "expected_updates": expected_count,
        "offered_updates": offered_count,
        "grouped_updates": len(by_update),
        "arrival_receipts": len(by_arrival),
        "completed_responses": len(all_responses),
        "successful_all_object_ttc": len(populations["successful_all_object_ttc"]),
        "failed_acknowledgements": len(populations["failed_acknowledgements"]),
        "updates_without_completion": len(by_update) - len(all_responses),
        "dropped_updates": sum(row.get("status") == "DROPPED_QUEUE_FULL" for row in arrivals),
        "source_errors": sum(row.get("status") == "SOURCE_ERROR" for row in arrivals),
        "missing_arrival_receipts": len(by_update.keys() - by_arrival.keys()),
        "unexpected_arrival_receipts": len(by_arrival.keys() - by_update.keys()),
    }
    return {
        "schema": "train40_paced_latency_populations_v1",
        "derivation": "grouped.updates.passed + arrival/object monotonic receipts",
        "quantile_method": "linear; index=(count-1)*fraction",
        "legacy_response_population": "ALL_RESPONSES_INCLUDING_FAILURES",
        "counts": counts,
        "counts_consistent": expected_count == offered_count == len(by_update)
        and not counts["unexpected_arrival_receipts"]
        and counts["completed_responses"] + counts["updates_without_completion"] == expected_count,
        "populations": {
            name: {
                "count": len(rows),
                **{
                    field: _distribution([row[field] for row in rows])
                    for field in (
                        "deadline_response_ms",
                        "raw_availability_response_ms",
                        "service_span_ms",
                    )
                },
            }
            for name, rows in populations.items()
        },
        "note": "Success is valid all-object output, not parity admission. "
        "Failed acknowledgements are not TTC completions. Drops have no inferred latency.",
    }
