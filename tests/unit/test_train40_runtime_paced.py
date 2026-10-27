"""Behavioral checks for independent arrivals, overload and full object coverage."""

from __future__ import annotations

import threading
import time

import pytest

from operational.train40_runtime.paced import run_paced_replay


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
