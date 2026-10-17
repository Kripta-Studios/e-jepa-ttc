"""CPU-only contract tests for prospective V19 cost v4."""

import copy

import pytest

from operational.decision_recovery.cost_measurement_v4 import (
    ARMS,
    BATCH,
    QUERY_COUNT,
    RAW_PHASE_ABS_ENVELOPE,
    TTC_ABS_ENVELOPE_SECONDS,
    _batch_indices,
    drift,
    protocol,
    summarize,
    validate_cold,
    validate_output,
    validate_row,
)


def _row(record, ordinal):
    index = record["indices"][ordinal]
    routes = {}
    for route in ("batch4_native", "batch1_descriptive"):
        routes[route] = {}
        for arm_number, arm in enumerate(ARMS):
            value = {
                "cached_P2_decode_s": 0.004,
                "host_to_device_s": 0.001,
                "forward_to_TTC_s": 0.002 + arm_number * 0.001,
                "cached_P2_to_TTC_s": 0.007 + arm_number * 0.001,
                "peak_extra_allocated_bytes": 0,
                "raw_phase": 0.01,
                "ttc_seconds": 10.0,
                "emitted_phase": 0.01,
            }
            if route == "batch1_descriptive":
                value["drift"] = drift(value, value)
            routes[route][arm] = value
    return {
        "ordinal": ordinal,
        "P_index": index,
        "order": record["paired_orders"][ordinal],
        "batch4_indices": _batch_indices(index),
        "routes": routes,
        "registration": "a" * 64,
        "attempt_id": "a" * 16,
        "utc": "2026-10-17T00:00:00+00:00",
    }


def test_protocol_reuses_fixed_queries_and_declares_nonexact_batch1():
    value = protocol()
    assert len(value["indices"]) == QUERY_COUNT
    assert value["historical_batch_size"] == BATCH
    assert value["optimizer_updates"] == 0 and value["backward"] is False
    assert "not asserted exact" in value["descriptive_route"]
    assert value["batch1_drift_contract"]["raw_phase_absolute_envelope"] == 1e-6
    assert value["batch1_drift_contract"]["ttc_absolute_envelope_seconds"] == 0.01
    assert value["batch1_drift_contract"]["saturation_rule"].startswith("abs(emitted TTC)")


def test_drift_uses_ttc_saturation_and_prospective_envelopes():
    expected = {"raw_phase": 0.0027904785, "ttc_seconds": 35.8861694, "emitted_phase": 0.0027904785}
    observed = {
        "raw_phase": expected["raw_phase"] + 3.22e-8,
        "ttc_seconds": expected["ttc_seconds"] + 0.000416,
        "emitted_phase": expected["emitted_phase"] + 3.22e-8,
    }
    evidence = drift(observed, expected)
    assert evidence["admitted"] is True
    assert evidence["observed_saturated"] is False
    changed = copy.deepcopy(observed)
    changed["raw_phase"] += RAW_PHASE_ABS_ENVELOPE * 2
    assert drift(changed, expected)["admitted"] is False
    changed = copy.deepcopy(observed)
    changed["ttc_seconds"] += TTC_ABS_ENVELOPE_SECONDS * 2
    assert drift(changed, expected)["admitted"] is False


def test_validate_output_requires_exact_batch4_and_guarded_batch1():
    expected = {"raw_phase": 0.01, "ttc_seconds": 10.0, "emitted_phase": 0.01}
    descriptive = {"raw_phase": 0.01000001, "ttc_seconds": 10.0001, "emitted_phase": 0.01000001}
    assert validate_output(expected, descriptive, expected)["admitted"] is True
    primary = dict(expected, raw_phase=0.01000001)
    with pytest.raises(ValueError, match="batch4"):
        validate_output(primary, descriptive, expected)


def test_rows_and_summary_are_complete_and_batch4_amortized():
    record = protocol()
    rows = [_row(record, ordinal) for ordinal in range(QUERY_COUNT)]
    for ordinal, row in enumerate(rows):
        validate_row(row, record, ordinal)
    result = summarize(rows)
    assert result["batch4_native"]["L0"]["forward_to_TTC"]["n"] == QUERY_COUNT
    assert result["batch4_native"]["L0"]["amortized_per_query"]["forward_to_TTC"][
        "mean_seconds"
    ] == pytest.approx(0.002 / 4)
    assert result["batch1_drift_max"]["LW"]["ttc_absolute_error_seconds"] == 0
    assert result["batch1_descriptive"]["L0"]["max_peak_extra_allocated_bytes"] == 0


def test_cold_validation_checks_output_and_memory():
    record = protocol()
    first = record["indices"][0]
    expected = {
        arm: {first: {"raw_phase": 0.01, "ttc_seconds": 10.0, "emitted_phase": 0.01}}
        for arm in ARMS
    }
    arm_value = {
        "checkpoint_load_eager_cpu_s": 0.1,
        "model_to_cuda_s": 0.1,
        "model_resident_delta_bytes": 1024,
        "reader_init_s": 0.1,
        "cached_P2_decode_s": 0.1,
        "host_to_device_s": 0.1,
        "first_forward_to_TTC_s": 0.1,
        "cached_P2_to_TTC_s": 0.3,
        "first_route_peak_extra_allocated_bytes": 2048,
        **expected["L0"][first],
    }
    cold = {
        "registration_sha256": "a" * 64,
        "attempt_id": "a" * 16,
        "meaning": "application-cold batch4 primary route; OS cache uncontrolled",
        "model_load_order": list(ARMS),
        "arms": {arm: copy.deepcopy(arm_value) for arm in ARMS},
        "optimizer_updates": 0,
        "backward": False,
    }
    validate_cold(cold, record, expected, "a" * 64, "a" * 16)
    cold["arms"]["LW"]["model_resident_delta_bytes"] = -1
    with pytest.raises(ValueError, match="cold timing"):
        validate_cold(cold, record, expected, "a" * 64, "a" * 16)
