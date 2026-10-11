"""Regression gates for NumPy observation identities and targeted queue recovery."""

import copy

import numpy as np
import pytest

from operational.rgb_port.infer_experts import _observation_signature
from operational.rgb_port_signature_fix.signature import observation_signature


def native_row():
    return {
        "producer_times_us": [2**53 + 1, 2**53 + 100001],
        "producer_sensor_times_us": [100, 200],
        "producer_available_us": [101, 201],
        "producer_shards": ["one.tar", "two.tar"],
        "producer_members": ["one.png", "two.png"],
        "producer_boxes_xyxy": [[1.5, 2.0, 30.0, 40.0], [2.0, 3.0, 31.0, 41.0]],
        "roi_xyxy": [1, 2, 30, 40],
    }


def test_native_hash_unchanged():
    row = native_row()
    assert observation_signature(row) == _observation_signature(row)


def test_numpy_nested_roundtrip_exact_and_non_mutating():
    row = native_row()
    arrays = {k: np.asarray(v) for k, v in row.items()}
    arrays["producer_boxes_xyxy"] = np.array(
        [np.array(v) for v in row["producer_boxes_xyxy"]], dtype=object
    )
    before = copy.deepcopy(arrays)
    with pytest.raises(TypeError, match="ndarray"):
        _observation_signature(arrays)
    assert observation_signature(arrays) == _observation_signature(row)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_integer_clocks_preserve_values_above_float_exactness():
    row = native_row()
    first = observation_signature(row)
    row["producer_times_us"] = np.asarray(row["producer_times_us"], dtype=np.int64)
    assert observation_signature(row) == first
    row["producer_times_us"][0] += 1
    assert observation_signature(row) != first


@pytest.mark.parametrize("field", list(native_row()))
def test_each_identity_field_still_detects_changes(field):
    row = native_row()
    expected = observation_signature(row)
    row[field] = ["changed"]
    assert observation_signature(row) != expected


def test_unused_labels_are_excluded():
    row = native_row()
    expected = observation_signature(row)
    row.update(target_ttc=np.nan, target_phase=object(), sample_token="different")
    assert observation_signature(row) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_identity_rejected(value):
    row = native_row()
    row["roi_xyxy"] = np.asarray([value])
    with pytest.raises(ValueError):
        observation_signature(row)


def test_numpy_scalars_tuples_and_zero_dimensional_arrays():
    row = {"roi_xyxy": (np.int64(1), np.float64(2.5), np.array(3), np.bool_(True))}
    assert observation_signature(row) == _observation_signature({"roi_xyxy": [1, 2.5, 3, True]})


def test_unsupported_values_are_not_stringified():
    with pytest.raises(TypeError):
        observation_signature({"roi_xyxy": object()})


@pytest.mark.parametrize(
    "module",
    ["operational.rgb_port.infer_experts", "operational.rgb_port_streaming.infer"],
)
def test_routing_survives_continuity_and_keeps_arguments(module):
    from operational.rgb_port_io_recovery.queue import route as io_route
    from operational.rgb_port_signature_fix.queue import route

    command = ["python", "-m", module, "observations", "--modality", "rgb"]
    expected = [*command]
    expected[2] = "operational.rgb_port_signature_fix.infer"
    assert route(command) == expected
    assert command[2] == module
    assert io_route(expected, None) == expected


def test_training_commands_unchanged():
    from operational.rgb_port_signature_fix.queue import route

    command = ["python", "-m", "operational.rgb_port.train_heads", "--fit-id", "R_CTX"]
    assert route(command) == command


def test_retry_only_failed_root_and_skipped_descendants():
    from operational.rgb_port_signature_fix.recovery import retry_state

    records = {
        "H_R_RAW": {"status": "FAILED", "attempts": 1, "reason": "child exit code 1"},
        "H_R_OBS": {"status": "SKIPPED_DEPENDENCY", "attempts": 0},
        "F_TRUE": {"status": "SKIPPED_DEPENDENCY", "attempts": 0},
        "E_CTX_MATCHED": {"status": "COMPLETE", "completed_updates": 2500},
        "UNRELATED": {"status": "FAILED", "reason": "different error"},
    }
    state = {"tasks": records, "campaign_status": "TERMINAL_WITH_BLOCKERS"}
    before = copy.deepcopy(state)
    tasks = [
        {"id": "H_R_RAW"},
        {"id": "H_R_OBS", "depends": ["H_R_RAW"]},
        {"id": "F_TRUE", "depends": ["H_R_OBS", "E_CTX_MATCHED"]},
        {"id": "E_CTX_MATCHED"},
        {"id": "UNRELATED"},
    ]
    result, changed = retry_state(state, tasks)
    assert state == before
    assert set(changed) == {"H_R_RAW", "H_R_OBS", "F_TRUE"}
    assert result["tasks"]["H_R_RAW"]["attempts"] == 1
    for name in ("E_CTX_MATCHED", "UNRELATED"):
        assert result["tasks"][name] == before["tasks"][name]


def test_retry_refuses_running_tasks_or_unrelated_root_state():
    from operational.rgb_port_signature_fix.recovery import retry_state

    for status in ("RUNNING", "COMPLETE", "WAITING"):
        with pytest.raises(ValueError):
            retry_state({"tasks": {"H_R_RAW": {"status": status}}}, [{"id": "H_R_RAW"}])
