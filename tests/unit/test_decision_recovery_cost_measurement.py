"""Prospective cost protocol and evidence validation without CUDA."""

import copy

import numpy as np
import pytest

from operational.decision_recovery.cost_measurement import (
    ARMS,
    PACK_ROWS,
    QUERY_COUNT,
    protocol,
    summarize,
    validate_cold,
    validate_outputs,
    validate_row,
)


def row(record, ordinal, scale=1.0):
    index = record["indices"][ordinal]
    return {
        "ordinal": ordinal,
        "P_index": index,
        "order": record["paired_orders"][ordinal],
        "cached_P2_decode_s": 0.004 * scale,
        "registration": "a" * 64,
        "attempt_id": "a" * 16,
        "utc": "2026-10-17T00:00:00+00:00",
        "timings": {
            arm: {
                "host_to_device_s": 0.001 * scale,
                "forward_to_TTC_s": (0.002 + arm_index * 0.001) * scale,
                "cached_P2_to_TTC_s": (0.007 + arm_index * 0.001) * scale,
                "ttc_seconds": 1.5 + arm_index,
                "emitted_phase": 0.1 + arm_index,
                "peak_extra_allocated_bytes": 1024 + arm_index,
                "memory_allocated_before_bytes": 2048 + arm_index,
            }
            for arm_index, arm in enumerate(ARMS)
        },
    }


def test_protocol_is_fixed_paired_batch1_and_zero_update():
    first, second = protocol(), protocol()
    assert first == second
    assert len(first["indices"]) == len(set(first["indices"])) == QUERY_COUNT
    assert all(0 <= index < PACK_ROWS for index in first["indices"])
    assert all(set(order) == set(ARMS) for order in first["paired_orders"])
    assert first["batch_size"] == 1 and first["warmups_per_endpoint"] == 20
    assert first["optimizer_updates"] == 0 and first["backward"] is False
    assert "cached native P2" in first["input_scope"]
    assert "raw event" in first["input_scope"]


def test_validate_and_summarize_complete_rows():
    record = protocol()
    rows = [row(record, ordinal, 1 + ordinal / 100) for ordinal in range(QUERY_COUNT)]
    for ordinal, value in enumerate(rows):
        validate_row(value, record, ordinal)
    result = summarize(rows)
    assert result["cached_P2_decode"]["n"] == QUERY_COUNT
    assert (
        result["L0"]["forward_to_TTC"]["p50_seconds"]
        < result["LW"]["forward_to_TTC"]["p50_seconds"]
    )
    assert result["LW"]["max_peak_extra_allocated_bytes"] == 1025
    assert result["order_counts"]["L0"] + result["order_counts"]["LW"] == QUERY_COUNT
    assert result["paired"]["forward_to_TTC"]["LW_over_L0_ratio"]["n"] == QUERY_COUNT


def test_durable_output_and_cold_parity_fail_closed():
    from operational.range_transfer.common import digest

    record = protocol()
    value = row(record, 0)
    index = value["P_index"]
    expected = {
        "L0": {index: (1.5, 0.1)},
        "LW": {index: (2.5, 1.1)},
    }
    validate_outputs(value, expected)
    changed = copy.deepcopy(value)
    changed["timings"]["LW"]["emitted_phase"] += 1
    with pytest.raises(ValueError, match="deployment output"):
        validate_outputs(changed, expected)

    cold = {}
    for arm in ARMS:
        ttc, phase = expected[arm][index]
        cold[arm] = {
            "torchscript_load_cpu_s": 0.1,
            "model_to_cuda_s": 0.1,
            "reader_init_s": 0.1,
            "cached_P2_decode_s": 0.1,
            "host_to_device_s": 0.1,
            "first_forward_to_TTC_s": 0.1,
            "cached_P2_to_TTC_s": 0.3,
            "ttc_seconds": ttc,
            "emitted_phase": phase,
            "model_resident_delta_bytes": 1024,
            "first_route_peak_allocated_bytes": 2048,
            "process_rss_bytes": 4096,
            "meaning": "application-cold; OS cache uncontrolled; descriptive single call",
        }
    payload = {
        "registration": digest(record),
        "registration_sha256": "a" * 64,
        "attempt_id": "a" * 16,
        "cold": cold,
        "cold_model_load_order": list(ARMS),
        "cold_memory_scope": record["cold_memory_limit"],
        "warmups_per_endpoint": 20,
        "optimizer_updates": 0,
    }
    validate_cold(payload, record, expected, "a" * 64, "a" * 16)
    payload["cold"]["L0"]["ttc_seconds"] = np.nan
    with pytest.raises(ValueError):
        validate_cold(payload, record, expected, "a" * 64, "a" * 16)


@pytest.mark.parametrize("corruption", ["ordinal", "index", "order", "nan", "missing"])
def test_row_corruption_fails_closed(corruption):
    record = protocol()
    value = row(record, 0)
    if corruption == "ordinal":
        value["ordinal"] = 1
    elif corruption == "index":
        value["P_index"] = record["indices"][1]
    elif corruption == "order":
        value["order"] = list(reversed(value["order"]))
    elif corruption == "nan":
        value["timings"]["L0"]["forward_to_TTC_s"] = np.nan
    else:
        del value["timings"]["LW"]["emitted_phase"]
    with pytest.raises(ValueError):
        validate_row(value, record, 0)


def test_summary_requires_all_rows():
    record = protocol()
    with pytest.raises(ValueError, match="All fixed"):
        summarize([copy.deepcopy(row(record, 0))])
