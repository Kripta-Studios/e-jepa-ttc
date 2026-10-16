"""CPU contract checks for optional transfer optimization and conservative fallback."""

import copy
from dataclasses import fields

import pytest
import torch

from operational.decision_recovery.runtime_performance import (
    BatchDeviceSensors,
    select_transfer_mode,
)
from operational.decision_recovery.training import validate_qa_proof
from operational.joint_measurement.geometry_capture import contiguous_subset
from operational.joint_measurement.qa import exact, fixture, subset


class CpuSensors:
    def __init__(self):
        self.data = fixture(16)
        self.population = 8
        self.requests = []

    def batch(self, ids):
        self.requests.append(list(ids))
        return subset(self.data, ids)


def test_one_batch_transport_preserves_all_fields_duplicates_rng_and_views():
    source = CpuSensors()
    wrapped = BatchDeviceSensors(source, device="cpu")
    ids = [1, 1, 3, 5, 0, 2, 6, 7] * 4
    state = torch.get_rng_state()
    result = wrapped.batch(ids)
    expected = subset(source.data, ids)
    assert source.requests == [ids]
    assert wrapped.population == 8
    assert torch.equal(state, torch.get_rng_state())
    for field in fields(expected):
        assert exact(getattr(result, field.name), getattr(expected, field.name)), field.name
    for i in range(0, 32, 4):
        micro = contiguous_subset(result, list(range(i, i + 4)))
        assert (
            micro.events.untyped_storage().data_ptr() == result.events.untyped_storage().data_ptr()
        )
        assert exact(micro.events, expected.events[i : i + 4])
    assert wrapped.counters()["device_batches"] == 1


def trials(ratio=0.9, peak=50):
    return [
        {"legacy_seconds": 1.0, "batch_seconds": ratio, "batch_peak_allocated_bytes": peak}
        for _ in range(5)
    ]


def test_transfer_admission_requires_measured_gain_and_memory():
    assert (
        select_transfer_mode(trials(), exact_candidate=True, device_total_bytes=100)[
            "selected_transfer"
        ]
        == "batch"
    )
    assert (
        select_transfer_mode(trials(0.99), exact_candidate=True, device_total_bytes=100)[
            "selected_transfer"
        ]
        == "legacy"
    )
    assert (
        select_transfer_mode(trials(0.9, 91), exact_candidate=True, device_total_bytes=100)[
            "selected_transfer"
        ]
        == "legacy"
    )


def test_first_capture_trial_is_excluded_from_speed_decision():
    rows = trials()
    rows[0]["batch_seconds"] = 100.0
    assert (
        select_transfer_mode(rows, exact_candidate=True, device_total_bytes=100)[
            "selected_transfer"
        ]
        == "batch"
    )


@pytest.mark.parametrize("mutation", ["inexact", "missing", "zero", "nan"])
def test_invalid_proofs_never_admit_transfer(mutation):
    rows = copy.deepcopy(trials())
    if mutation == "missing":
        rows.pop()
    elif mutation == "zero":
        rows[2]["legacy_seconds"] = 0
    elif mutation == "nan":
        rows[2]["batch_seconds"] = float("nan")
    with pytest.raises(ValueError):
        select_transfer_mode(rows, exact_candidate=mutation != "inexact", device_total_bytes=100)


def qa_proof():
    checks = {
        "L0_exact_native_masses": True,
        "CPU_native_transport_duplicate_draw": True,
        "CPU_transport_preserves_rng": True,
    }
    for arm in ("L0", "LW"):
        for name in (
            "zero_update_weights_1",
            "zero_update_weights_2",
            "native_loss_gradient_rng_exact",
            "persistent_single_capture",
            "persistent_single_loss_capture",
            "persistent_complete_state",
            "persistent_curve",
            "resumed_complete_state",
            "resumed_curve",
        ):
            checks[f"{arm}_{name}"] = True
    performance = select_transfer_mode(trials(), exact_candidate=True, device_total_bytes=100)
    performance.update(per_trial_exact=[True] * 5, weights_unchanged=True)
    return {
        "status": "PASS",
        "registration_sha256": "registered",
        "source_commit": "source",
        "physical_technical_updates": 24,
        "scientific_updates": 0,
        "accounting": {"scientific": 0, "technical": 24, "recovery_upper": 0, "pending_upper": 0},
        "checks": checks,
        "runtime": {
            "selected_transfer": "batch",
            "per_arm": {
                "L0": copy.deepcopy(performance),
                "LW": copy.deepcopy(performance),
            },
        },
    }


def test_complete_qa_proof_admits_recomputed_mode():
    assert validate_qa_proof(qa_proof(), {"source_commit": "source"}, "registered") == "batch"


@pytest.mark.parametrize(
    "mutation", ["false_resume", "missing_check", "count", "commit", "mode", "speed", "weights"]
)
def test_qa_saved_pass_does_not_override_actual_contract(mutation):
    proof = qa_proof()
    if mutation == "false_resume":
        proof["checks"]["L0_resumed_complete_state"] = False
    elif mutation == "missing_check":
        proof["checks"].pop("LW_persistent_single_loss_capture")
    elif mutation == "count":
        proof["physical_technical_updates"] = 23
    elif mutation == "commit":
        proof["source_commit"] = "unrelated"
    elif mutation == "mode":
        proof["runtime"]["selected_transfer"] = "legacy"
    elif mutation == "speed":
        proof["runtime"]["per_arm"]["L0"]["warm_median_paired_ratio"] = 0.5
    else:
        proof["runtime"]["per_arm"]["LW"]["weights_unchanged"] = False
    with pytest.raises(ValueError):
        validate_qa_proof(proof, {"source_commit": "source"}, "registered")
