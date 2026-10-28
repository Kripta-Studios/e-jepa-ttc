"""Behavioral checks for independent arrivals, overload and full object coverage."""

from __future__ import annotations

import threading
import time

import pytest

from operational.train40_runtime.paced import (
    pacing_clock_metadata,
    run_paced_replay,
    wait_until_deadline,
)
from operational.train40_runtime.paced_statistics import paced_latency_populations


def prediction(value=1.0):
    return {
        "success": True,
        "predictions": {
            "PHASE_ONLY_TRAIN40_S7": value,
            "E_CTX_TRAIN40": value,
            "R_CTX_TRAIN40": value,
        },
    }


def test_raw_producer_runs_independently_and_preserves_all_object_identities():
    main_id = threading.get_ident()
    source_ids, compute_ids = [], []

    def acquire(index):
        source_ids.append(threading.get_ident())
        return index

    def infer(arrival):
        compute_ids.append(threading.get_ident())
        return {key: prediction(arrival.payload + 1) for key in ("a", "b")}

    # This test checks ownership/coverage, not host scheduling under contention.
    # A separate overload test intentionally bounds the queue below population.
    result = run_paced_replay(acquire, infer, ["a", "b"], updates=3, rate_hz=200, queue_capacity=3)
    assert result["grouped"]["passed"]
    assert result["grouped"]["expected_object_results"] == 6
    assert set(source_ids) != {main_id}
    assert set(compute_ids) == {main_id}
    assert result["physical_goal_demonstrated"] is False
    assert not result["capacity_checks_passed"]
    populations = result["latency_populations"]
    assert populations["counts_consistent"]
    assert populations["counts"]["successful_all_object_ttc"] == 3
    assert populations["counts"]["failed_acknowledgements"] == 0
    arrivals = result["arrival_receipts"]
    assert arrivals[2]["scheduled_ns"] - arrivals[0]["scheduled_ns"] == 10_000_000
    for row in result["object_records"]:
        assert row["scheduled_ns"] <= row["observation_available_ns"]
        assert row["observation_available_ns"] <= row["started_ns"] <= row["completed_ns"]


def test_overload_keeps_fixed_schedule_and_records_drops_without_postponing_arrivals():
    fourth_arrival = threading.Event()

    def acquire(index):
        if index == 4:
            fourth_arrival.set()
        return index

    def infer(arrival):
        assert fourth_arrival.wait(2.0)
        return {"a": prediction()}

    result = run_paced_replay(acquire, infer, ["a"], updates=7, rate_hz=500, queue_capacity=1)
    assert result["dropped_updates"] > 0
    assert len(result["arrival_receipts"]) == 7
    assert not result["grouped"]["passed"]
    assert not result["capacity_checks"]["no_drops_or_source_errors"]
    assert (
        result["arrival_receipts"][-1]["scheduled_ns"]
        - result["arrival_receipts"][0]["scheduled_ns"]
        == 12_000_000
    )


@pytest.mark.parametrize("failure", ["source", "inference", "missing_object", "unexpected_object"])
def test_failures_are_retained_and_cannot_pass_capacity(failure):
    def acquire(index):
        if failure == "source" and index == 1:
            raise RuntimeError("packet missing")
        return index

    def infer(arrival):
        if failure == "inference":
            raise RuntimeError("guard rejected")
        result = {"a": prediction(), "b": prediction()}
        if failure == "missing_object":
            result.pop("b")
        if failure == "unexpected_object":
            result["stranger"] = prediction()
        return result

    result = run_paced_replay(acquire, infer, ["a", "b"], updates=3, rate_hz=200)
    assert not result["grouped"]["passed"]
    assert not result["capacity_checks_passed"]
    if failure == "source":
        assert result["source_errors"] == 1
        assert result["arrival_receipts"][1]["status"] == "SOURCE_ERROR"


def test_source_delay_is_visible_and_does_not_reset_the_schedule():
    def acquire(index):
        time.sleep(0.005)
        return index

    result = run_paced_replay(
        acquire, lambda item: {"a": prediction()}, ["a"], updates=2, rate_hz=500
    )
    assert result["raw_availability_lateness_ms"]["min_ms"] >= 4.0
    assert (
        result["deadline_response_ms"]["mean_ms"]
        > result["raw_availability_response_ms"]["mean_ms"]
    )
    receipts = result["arrival_receipts"]
    assert receipts[1]["scheduled_ns"] - receipts[0]["scheduled_ns"] == 2_000_000


@pytest.mark.parametrize("ids", [[], ["a", "a"], [""], [1]])
def test_invalid_object_population_is_rejected(ids):
    with pytest.raises(ValueError, match="object_ids"):
        run_paced_replay(lambda index: index, lambda item: {}, ids, updates=2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True, "1.0"])
def test_success_flag_cannot_hide_invalid_ttc(value):
    result = run_paced_replay(
        lambda index: index,
        lambda item: {"a": prediction(value)},
        ["a"],
        updates=1,
        startup_lead_s=0,
    )
    assert not result["grouped"]["passed"]


def test_success_without_predictions_is_rejected():
    result = run_paced_replay(
        lambda index: index,
        lambda item: {"a": {"success": True}},
        ["a"],
        updates=1,
        startup_lead_s=0,
    )
    assert not result["grouped"]["passed"]
    populations = result["latency_populations"]
    assert populations["counts"]["successful_all_object_ttc"] == 0
    assert populations["counts"]["failed_acknowledgements"] == 1
    assert populations["populations"]["successful_all_object_ttc"]["deadline_response_ms"] is None


def test_failure_acknowledgements_do_not_reduce_valid_ttc_percentiles():
    arrivals, updates = [], []
    for index, (passed, duration) in enumerate(((True, 30), (False, 1), (False, 2))):
        scheduled = 1_000_000_000 + index * 100_000_000
        arrivals.append(
            {
                "update_id": str(index),
                "scheduled_ns": scheduled,
                "observation_available_ns": scheduled + 500_000,
                "status": "ENQUEUED",
            }
        )
        updates.append(
            {
                "update_id": str(index),
                "passed": passed,
                "coverage": {"passed": True},
                "started_ns": scheduled + 500_000,
                "completed_ns": scheduled + duration * 1_000_000,
            }
        )
    arrivals.append({"update_id": "3", "status": "DROPPED_QUEUE_FULL"})
    updates.extend([{"update_id": "3", "passed": False}, {"update_id": "4", "passed": False}])
    result = paced_latency_populations(
        {
            "offered_updates": 5,
            "arrival_receipts": arrivals,
            "grouped": {"expected_updates": 5, "passed": False, "updates": updates},
        }
    )
    assert result is not None and result["counts_consistent"]
    assert result["counts"]["dropped_updates"] == result["counts"]["missing_arrival_receipts"] == 1
    assert result["counts"]["updates_without_completion"] == 2
    populations = result["populations"]
    assert populations["successful_all_object_ttc"]["deadline_response_ms"]["p50_ms"] == 30
    assert populations["failed_acknowledgements"]["deadline_response_ms"]["p50_ms"] == 1.5
    assert populations["failed_acknowledgements"]["deadline_response_ms"]["p95_ms"] == 1.95
    assert populations["all_responses_including_failures"]["deadline_response_ms"]["p50_ms"] == 2


def test_success_population_rejects_contradictory_coverage():
    with pytest.raises(ValueError, match="incomplete coverage"):
        paced_latency_populations(
            {
                "arrival_receipts": [
                    {
                        "update_id": "a",
                        "status": "ENQUEUED",
                        "scheduled_ns": 0,
                        "observation_available_ns": 0,
                    }
                ],
                "grouped": {
                    "updates": [
                        {
                            "update_id": "a",
                            "passed": True,
                            "started_ns": 0,
                            "completed_ns": 1,
                            "coverage": {"passed": False},
                        }
                    ]
                },
            }
        )


def test_sleep_wait_uses_absolute_remainder_without_busy_spin():
    stop = threading.Event()
    now_ns = 0
    waits = []

    def sleep(seconds):
        nonlocal now_ns
        waits.append(seconds)
        now_ns += round(seconds * 1e9)

    cancelled = wait_until_deadline(
        25_000_000, stop, wait_strategy="sleep", clock_ns=lambda: now_ns, sleep=sleep
    )
    assert not cancelled
    assert waits == pytest.approx([0.010, 0.010, 0.005])
    assert now_ns == 25_000_000


def test_sleep_overshoot_does_not_shift_the_next_absolute_deadline():
    now_ns = 0
    waits = []

    def sleep(seconds):
        nonlocal now_ns
        waits.append(seconds)
        now_ns += round(seconds * 1e9) + (6_000_000 if len(waits) == 1 else 0)

    stop = threading.Event()
    assert not wait_until_deadline(
        20_000_000, stop, wait_strategy="sleep", clock_ns=lambda: now_ns, sleep=sleep
    )
    assert not wait_until_deadline(
        40_000_000, stop, wait_strategy="sleep", clock_ns=lambda: now_ns, sleep=sleep
    )
    assert waits == pytest.approx([0.010, 0.004, 0.010, 0.010])
    assert now_ns == 40_000_000


@pytest.mark.parametrize("already_stopped", [False, True])
def test_sleep_cancellation_is_checked_between_bounded_waits(already_stopped):
    stop = threading.Event()
    waits = []
    if already_stopped:
        stop.set()

    def sleep(seconds):
        waits.append(seconds)
        stop.set()

    assert wait_until_deadline(10**9, stop, wait_strategy="sleep", clock_ns=lambda: 0, sleep=sleep)
    assert waits == ([] if already_stopped else [0.010])


@pytest.mark.parametrize("deadline_ns", [0, -1, -1_000_000])
def test_sleep_past_deadline_does_not_issue_a_zero_or_negative_wait(deadline_ns):
    assert not wait_until_deadline(
        deadline_ns,
        threading.Event(),
        wait_strategy="sleep",
        clock_ns=lambda: 0,
        sleep=lambda seconds: pytest.fail("Late arrivals must not sleep or spin"),
    )


def test_event_wait_preserves_original_timeout_and_cancellation_result():
    class Event(threading.Event):
        def __init__(self):
            super().__init__()
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return True

    stop = Event()
    assert wait_until_deadline(30_000_000, stop, wait_strategy="event", clock_ns=lambda: 10_000_000)
    assert wait_until_deadline(5_000_000, stop, wait_strategy="event", clock_ns=lambda: 10_000_000)
    assert stop.waits == [0.020, 0.0]


@pytest.mark.parametrize("strategy", ["event", "sleep"])
def test_strategy_metadata_and_same_deadline_grid_are_preserved(strategy):
    result = run_paced_replay(
        lambda index: index,
        lambda arrival: {"a": prediction()},
        ["a"],
        updates=3,
        rate_hz=100,
        queue_capacity=3,
        wait_strategy=strategy,
    )
    receipts = result["arrival_receipts"]
    assert result["grouped"]["passed"]
    assert [row["scheduled_ns"] - receipts[0]["scheduled_ns"] for row in receipts] == [
        0,
        10_000_000,
        20_000_000,
    ]
    assert result["wait_strategy"] == strategy
    metadata = result["scheduler"]
    assert metadata["clock_implementation"]
    assert metadata["python_version"]
    assert metadata["deadlines_absolute"] and not metadata["busy_spin"]
    assert metadata["scheduler_lateness_subtracted"] is False
    assert metadata["sleep_stop_poll_seconds"] == (0.010 if strategy == "sleep" else None)


def test_invalid_wait_strategy_fails_before_any_acquisition():
    with pytest.raises(ValueError, match="wait_strategy"):
        run_paced_replay(
            lambda index: pytest.fail("Acquisition started"),
            lambda item: {},
            ["a"],
            updates=1,
            wait_strategy="spin",
        )
    with pytest.raises(ValueError, match="wait_strategy"):
        pacing_clock_metadata("spin")


def test_default_wait_strategy_is_still_legacy_event():
    result = run_paced_replay(
        lambda index: index, lambda item: {"a": prediction()}, ["a"], updates=1, startup_lead_s=0
    )
    assert result["wait_strategy"] == "event"


def test_source_lifecycle_owns_one_thread_and_starts_deadlines_after_warmup():
    main = threading.get_ident()
    calls = []
    warmup_end = []

    def warmup():
        calls.append(("warmup", threading.get_ident()))
        warmup_end.append(time.perf_counter_ns())
        return {"causal_prefill": True}

    def acquire(index):
        calls.append(("acquire", threading.get_ident()))
        return index

    def finalize():
        calls.append(("finalize", threading.get_ident()))
        return {"closed": True}

    result = run_paced_replay(
        acquire,
        lambda item: {"a": prediction()},
        ["a"],
        updates=2,
        rate_hz=100,
        wait_strategy="sleep",
        source_warmup=warmup,
        source_finalize=finalize,
    )
    assert [name for name, _ in calls] == ["warmup", "acquire", "acquire", "finalize"]
    assert len({identity for _, identity in calls}) == 1
    assert calls[0][1] != main
    assert result["first_scheduled_ns"] >= warmup_end[0] + 50_000_000
    assert result["source_lifecycle"]["warmup"]["result"] == {"causal_prefill": True}
    assert result["source_lifecycle"]["finalize"]["result"] == {"closed": True}
    assert result["capacity_checks"]["source_lifecycle_succeeded"]


@pytest.mark.parametrize("failure", ["warmup", "acquire", "finalize"])
def test_lifecycle_finalization_survives_source_failures(failure):
    calls = []

    def warmup():
        calls.append("warmup")
        if failure == "warmup":
            raise RuntimeError("cannot open")

    def acquire(index):
        calls.append("acquire")
        if failure == "acquire":
            raise RuntimeError("cannot read")
        return index

    def finalize():
        calls.append("finalize")
        if failure == "finalize":
            raise RuntimeError("cannot close")

    result = run_paced_replay(
        acquire,
        lambda item: {"a": prediction()},
        ["a"],
        updates=1,
        startup_lead_s=0,
        source_warmup=warmup,
        source_finalize=finalize,
    )
    assert calls[-1] == "finalize"
    assert not result["capacity_checks_passed"]
    if failure == "warmup":
        assert calls == ["warmup", "finalize"]
        assert result["missing_arrival_ids"] == ["update-0"]
    if failure == "acquire":
        assert result["source_errors"] == 1
    else:
        assert not result["capacity_checks"]["source_lifecycle_succeeded"]
        assert result["source_lifecycle"]["errors"][0]["phase"] == failure


def test_caller_cancellation_still_finalizes_source_on_producer_thread():
    ids = {}

    class Cancelled(BaseException):
        pass

    def acquire(index):
        ids["acquire"] = threading.get_ident()
        return index

    def infer(arrival):
        raise Cancelled()

    def finalize():
        ids["finalize"] = threading.get_ident()

    with pytest.raises(Cancelled):
        run_paced_replay(
            acquire,
            infer,
            ["a"],
            updates=1000,
            startup_lead_s=0,
            wait_strategy="sleep",
            source_finalize=finalize,
        )
    assert ids["acquire"] == ids["finalize"] != threading.get_ident()
