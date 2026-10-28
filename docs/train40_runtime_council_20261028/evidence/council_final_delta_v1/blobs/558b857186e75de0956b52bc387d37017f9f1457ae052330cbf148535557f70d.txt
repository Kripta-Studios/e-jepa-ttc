"""Latency gates reject partial populations, hidden queueing and changed outputs."""

from copy import deepcopy

import pytest

from operational.train40_runtime.measurement import (
    PREDICTION_COMPONENTS,
    coverage_report,
    group_updates,
    latency_distribution,
    prediction_parity,
    scheduled_stream_report,
    snapshot_files,
    verify_snapshot,
)


def prediction(query_id="q1"):
    return {
        "query_id": query_id,
        "predictions": dict.fromkeys(PREDICTION_COMPONENTS, 2.0),
        "rgb_available": True,
        "rgb_unavailable_reason": "",
        "history_lengths": {"event": 4, "rgb": 5},
    }


def scope():
    return {
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


def object_records(count=5, service_ns=12_000_000):
    records, expected = [], {}
    for index in range(count):
        update = f"u{index}"
        expected[update] = ["car", "pedestrian"]
        for object_id, offset in (("car", 0), ("pedestrian", service_ns)):
            scheduled = index * 20_000_000
            records.append(
                {
                    "update_id": update,
                    "object_id": object_id,
                    "scheduled_ns": scheduled,
                    "started_ns": scheduled + offset,
                    "completed_ns": scheduled + offset + service_ns,
                    "success": True,
                }
            )
    return records, expected


def test_coverage_preserves_missing_duplicate_and_extra_ids():
    result = coverage_report(["a", "b"], [{"query_id": "a"}, {"query_id": "a"}, {"query_id": "x"}])
    assert not result["passed"]
    assert result["missing_ids"] == ["b"]
    assert result["duplicate_ids"] == ["a"]
    assert result["unexpected_ids"] == ["x"]
    with pytest.raises(ValueError, match="duplicate"):
        coverage_report(["a", "a"], [])


def test_parity_checks_every_component_fixed_tolerance_and_exactness():
    old = [prediction()]
    new = deepcopy(old)
    exact = prediction_parity(old, new, ["q1"])
    assert exact["passed"] and exact["exact"]
    new[0]["predictions"]["PHASE_ONLY_TRAIN40_S7"] += 2e-5
    close = prediction_parity(old, new, ["q1"])
    assert close["passed"] and not close["exact"]
    new[0]["predictions"]["R_CTX_TRAIN40"] += 1e-3
    failed = prediction_parity(old, new, ["q1"])
    assert not failed["passed"]
    assert failed["components"]["R_CTX_TRAIN40"]["max_abs_error"] == pytest.approx(1e-3)
    assert failed["atol"] == failed["rtol"] == 1e-5


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "2.0", True])
def test_parity_rejects_invalid_final_output_even_when_both_sides_match(value):
    old = prediction()
    old["predictions"]["PHASE_ONLY_TRAIN40_S7"] = value
    assert not prediction_parity([old], [deepcopy(old)], ["q1"])["passed"]


def test_parity_requires_causal_metadata_and_finite_mask():
    old, new = prediction(), prediction()
    new["history_lengths"]["rgb"] = 4
    assert not prediction_parity([old], [new], ["q1"])["passed"]
    del new["rgb_unavailable_reason"]
    result = prediction_parity([old], [new], ["q1"])
    assert "rgb_unavailable_reason" in result["per_query"][0]["metadata_mismatches"]
    new = prediction()
    new["predictions"]["R_CTX_TRAIN40"] = None
    result = prediction_parity([old], [new], ["q1"])
    assert not result["per_query"][0]["components"]["R_CTX_TRAIN40"]["finite_mask_equal"]
    assert not result["passed"]


def test_event_only_fallback_must_be_explicit_and_unchanged():
    old = prediction()
    old.update(rgb_available=False, rgb_unavailable_reason="NO_CAUSAL_RGB")
    old["history_lengths"]["rgb"] = 0
    old["predictions"]["R_CTX_TRAIN40"] = None
    assert prediction_parity([old], [deepcopy(old)], ["q1"])["passed"]
    assert not prediction_parity([old], [], ["q1"])["passed"]


def test_latency_distribution_reports_linear_tails_and_does_not_claim_hz():
    result = latency_distribution([10.0, 10.0, 10.0, 110.0])
    assert result["mean_ms"] == 35
    assert result["p95_ms"] == pytest.approx(95)
    assert result["p99_ms"] == pytest.approx(107)
    assert result["over_100ms_count"] == 1
    assert "sustained_hz" not in result
    with pytest.raises(ValueError):
        latency_distribution([float("nan")])


def test_update_completion_uses_last_object_and_never_divides_by_count():
    records, expected = object_records()
    grouped = group_updates(records, expected)
    assert grouped["passed"]
    assert grouped["updates"][0]["response_ms"] == 24.0
    report = scheduled_stream_report(
        grouped, scope(), minimum_duration_s=0.1, required_objects_per_update=2
    )
    assert not report["passed"]
    assert not report["gates"]["every_complete_update_strictly_under_20ms"]
    assert report["max_outstanding_before_arrival"] == 1


def test_grouping_preserves_object_failure_missing_objects_and_extra_updates():
    records, expected = object_records()
    records.pop()
    records[0]["success"] = False
    records.append({**records[1], "update_id": "unexpected"})
    grouped = group_updates(records, expected)
    assert not grouped["passed"]
    assert grouped["unexpected_update_ids"] == ["unexpected"]
    assert grouped["updates"][-1]["coverage"]["missing_ids"] == ["pedestrian"]
    result = scheduled_stream_report(grouped, scope(), minimum_duration_s=0.1)
    assert result["timed_update_count"] == 5
    assert result["successful_complete_update_count"] == 3


def test_fast_control_data_passes_logic_but_primary_replay_cannot_certify_physical_goal():
    records, expected = object_records(service_ns=4_000_000)
    grouped = group_updates(records, expected)
    report = scheduled_stream_report(
        grouped, scope(), minimum_duration_s=0.1, required_objects_per_update=2
    )
    assert report["passed"]
    assert report["observed_completed_updates_per_wall_second"] == 50
    primary = scope()
    primary["primary_only_cohort"] = True
    primary["paced_physical_arrivals"] = False
    rejected = scheduled_stream_report(grouped, primary, minimum_duration_s=0.1)
    assert not rejected["passed"]
    assert set(rejected["scope_failures"]) == {"primary_only_cohort", "paced_physical_arrivals"}
    assert not scheduled_stream_report(grouped, scope())["gates"]["sustained_observation_duration"]


def test_queue_wait_is_part_of_response_and_observed_throughput():
    records, expected = object_records(service_ns=4_000_000)
    for row in records:
        row["started_ns"] += 100_000_000
        row["completed_ns"] += 100_000_000
    result = scheduled_stream_report(
        group_updates(records, expected), scope(), minimum_duration_s=0.1
    )
    assert result["response"]["mean_ms"] == 108
    assert result["service_span"]["mean_ms"] == 8
    assert result["queue_wait"]["mean_ms"] == 100
    assert result["observed_completed_updates_per_wall_second"] < 50
    assert result["pending_after_last_period"] == 5
    assert not result["passed"]


def test_schedule_must_be_fixed_without_resetting_after_slow_completions():
    records, expected = object_records(service_ns=4_000_000)
    for index, row in enumerate(records):
        for name in ("scheduled_ns", "started_ns", "completed_ns"):
            row[name] += (index // 2) * 8_000_000
    result = scheduled_stream_report(
        group_updates(records, expected), scope(), minimum_duration_s=0.1
    )
    assert not result["gates"]["paced_50hz"]
    assert not result["passed"]


def test_twenty_ms_is_not_strictly_below_twenty():
    records, expected = object_records(service_ns=10_000_000)
    result = scheduled_stream_report(
        group_updates(records, expected), scope(), minimum_duration_s=0.1
    )
    assert result["gates"]["p95_at_most_50ms"]
    assert not result["gates"]["every_complete_update_strictly_under_20ms"]


def capacity_records(object_count=20):
    records, expected = [], {}
    for index in range(5):
        identity = f"u{index}"
        expected[identity] = [f"object{number}" for number in range(object_count)]
        for number, object_id in enumerate(expected[identity]):
            release = index * 20_000_000
            records.append(
                {
                    "update_id": identity,
                    "object_id": object_id,
                    "scheduled_ns": release,
                    "started_ns": release + number * 100_000,
                    "completed_ns": release + (number + 1) * 100_000,
                    "success": True,
                }
            )
    return records, expected


@pytest.mark.parametrize("object_count", [1, 19, 20, 21])
def test_default_task_gate_requires_exactly_twenty_distinct_objects(object_count):
    records, expected = capacity_records(object_count)
    grouped = group_updates(records, expected)
    assert grouped["passed"]
    report = scheduled_stream_report(grouped, scope(), minimum_duration_s=0.1)
    assert report["required_objects_per_update"] == 20
    assert report["passed"] is (object_count == 20)
    assert report["gates"]["required_distinct_objects_per_update"] is (object_count == 20)
    assert report["object_capacity"]["updates_meeting_capacity"] == (5 if object_count == 20 else 0)
    assert report["object_capacity"]["per_update"][0]["distinct_successful_objects"] == object_count


@pytest.mark.parametrize("damage", ["missing", "duplicate", "failure", "wrong_identity"])
def test_twenty_object_capacity_rejects_lost_duplicate_or_failed_completions(damage):
    records, expected = capacity_records()
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records[-1]["object_id"] = records[-2]["object_id"]
    elif damage == "failure":
        records[-1]["success"] = False
    else:
        records[-1]["object_id"] = "unexpected_object"
    report = scheduled_stream_report(
        group_updates(records, expected), scope(), minimum_duration_s=0.1
    )
    assert not report["passed"]
    assert report["object_capacity"]["updates_meeting_capacity"] == 4
    assert not report["object_capacity"]["per_update"][-1]["passed"]


def test_claimed_object_count_without_actual_identities_cannot_pass():
    records, expected = capacity_records(1)
    grouped = group_updates(records, expected)
    for row in grouped["updates"]:
        row["objects"] = 20
    report = scheduled_stream_report(grouped, scope(), minimum_duration_s=0.1)
    assert not report["passed"]
    assert not report["object_capacity"]["passed"]


@pytest.mark.parametrize("invalid", [0, -1, True, 20.0])
def test_required_object_capacity_must_be_a_positive_integer(invalid):
    records, expected = capacity_records()
    with pytest.raises(ValueError, match="positive integer"):
        scheduled_stream_report(
            group_updates(records, expected), scope(), required_objects_per_update=invalid
        )


def test_inventory_detects_changes_and_deletion_without_rewriting(tmp_path):
    first, second = tmp_path / "weight.pt", tmp_path / "predictions.json"
    first.write_bytes(b"frozen weights")
    second.write_bytes(b"original predictions")
    snapshot = snapshot_files([first, second])
    assert verify_snapshot(snapshot)["passed"]
    first.write_bytes(b"changed weights")
    second.unlink()
    result = verify_snapshot(snapshot)
    assert not result["passed"]
    assert result["changed"] == [str(first.resolve())]
    assert result["missing"] == [str(second.resolve())]
