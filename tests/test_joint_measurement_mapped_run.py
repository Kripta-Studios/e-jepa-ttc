"""No I/O optimization may bypass exact-state or measured-speed evidence."""

import copy

import pytest

from operational.joint_measurement.mapped_run import validate_qa


def result() -> dict:
    keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "numpy_random_state",
        "python_random_state",
        "cursor",
    )
    return {
        "status": "PASS",
        "performance_gate_pass": True,
        "optimizer_updates": 80,
        "checks": {
            fit: {
                "status": "PASS",
                "exact": {key: True for key in keys},
                "mapped_to_serial_ratio": 0.6,
            }
            for fit in ["A", "B", "C", "D"]
        },
    }


def test_full_exact_and_speed_evidence_required():
    evidence = result()
    validate_qa(evidence, ["A", "B", "C", "D"])
    for field, value in [
        ("status", "FAIL"),
        ("performance_gate_pass", False),
        ("optimizer_updates", 79),
    ]:
        changed = {**evidence, field: value}
        with pytest.raises(ValueError):
            validate_qa(changed, ["A", "B", "C", "D"])
    with pytest.raises(ValueError):
        validate_qa(evidence, ["A", "B", "C", "D", "missing"])


@pytest.mark.parametrize(
    "alteration", ["missing_cursor", "unequal_weights", "slow_one", "slow_mean"]
)
def test_claimed_pass_cannot_hide_underlying_failures(alteration):
    evidence = copy.deepcopy(result())
    if alteration == "missing_cursor":
        del evidence["checks"]["A"]["exact"]["cursor"]
    elif alteration == "unequal_weights":
        evidence["checks"]["A"]["exact"]["model_state_dict"] = False
    elif alteration == "slow_one":
        evidence["checks"]["A"]["mapped_to_serial_ratio"] = 1.1
    else:
        for check in evidence["checks"].values():
            check["mapped_to_serial_ratio"] = 0.85
    with pytest.raises(ValueError):
        validate_qa(evidence, ["A", "B", "C", "D"])
