"""Missing or failing evidence must prevent CUDA runtime admission."""

import copy

import pytest

from operational.joint_measurement.geometry_capture_qa import STATE_KEYS
from operational.joint_measurement.geometry_capture_run import validate_proofs


def proofs():
    probe = {
        "status": "CANDIDATE_FOR_RESUME_QA",
        "checks": {
            name: True
            for name in (
                "losses",
                "gradients",
                "rng_exact",
                "weights_unchanged",
                "pinned_microbatch_views",
                "parameter_identity",
            )
        },
        "ratio": 0.3,
    }
    qa = {
        "status": "PASS",
        "exact_graph_resume": {name: True for name in (*STATE_KEYS, "curve")},
        "eager_numerical_comparison": {name: True for name in (*STATE_KEYS, "curve")},
    }
    return probe, qa


def test_complete_runtime_proofs_are_admissible():
    validate_proofs(*proofs())


@pytest.mark.parametrize("section", ["exact_graph_resume", "eager_numerical_comparison"])
@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_full_state_proof_rejected(section, missing):
    probe, qa = proofs()
    if missing:
        del qa[section]["optimizer_state_dict"]
    else:
        qa[section]["optimizer_state_dict"] = False
    with pytest.raises(ValueError):
        validate_proofs(probe, qa)


def test_missing_gradient_or_slow_capture_does_not_admit():
    probe, qa = proofs()
    without = copy.deepcopy(probe)
    del without["checks"]["gradients"]
    with pytest.raises(ValueError):
        validate_proofs(without, qa)
    probe["ratio"] = 0.99
    with pytest.raises(ValueError):
        validate_proofs(probe, qa)
