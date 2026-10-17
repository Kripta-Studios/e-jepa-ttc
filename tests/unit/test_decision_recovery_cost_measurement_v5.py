"""CPU-only semantic checks for V19 cost V5."""

import copy

import numpy as np
import pytest

from operational.decision_recovery.cost_measurement_v5 import (
    ARMS,
    _validate_cuda_reference,
    protocol,
    validate_primary,
)


def test_protocol_changes_only_derived_phase_reference_semantics():
    value = protocol()
    assert value["schema"].endswith("cost_v5")
    assert "CUDA reference" in value["emitted_phase_contract"]
    assert value["batch1_drift_contract"]["raw_phase_absolute_envelope"] == 1e-6
    assert value["batch1_drift_contract"]["ttc_absolute_envelope_seconds"] == 0.01
    assert value["optimizer_updates"] == 0 and value["backward"] is False


def test_primary_keeps_raw_and_ttc_exact_and_uses_cuda_derived_phase():
    expected = {"raw_phase": -0.021, "ttc_seconds": -4.68, "emitted_phase": -0.020999998}
    observed = {"raw_phase": -0.021, "ttc_seconds": -4.68, "emitted_phase": -0.021}
    validate_primary(observed, expected, -0.021)
    with pytest.raises(ValueError, match="raw phase"):
        validate_primary(dict(observed, raw_phase=-0.0209999), expected, -0.021)
    with pytest.raises(ValueError, match="TTC"):
        validate_primary(dict(observed, ttc_seconds=-4.680001), expected, -0.021)
    with pytest.raises(ValueError, match="CUDA emitted"):
        validate_primary(dict(observed, emitted_phase=-0.0209999), expected, -0.021)


def test_cuda_reference_requires_finite_values_and_exact_ttc_binding():
    record = {"indices": [7]}
    expected = {
        arm: {7: {"raw_phase": 0.01, "ttc_seconds": 10.0, "emitted_phase": 0.01}} for arm in ARMS
    }
    payload = {
        "registration_sha256": "a" * 64,
        "backend": "CUDA",
        "values": {
            arm: {"7": {"input_ttc_seconds": 10.0, "emitted_phase_cuda": 0.01}} for arm in ARMS
        },
        "optimizer_updates": 0,
        "backward": False,
    }
    assert _validate_cuda_reference(payload, expected, record, "a" * 64)["L0"][7] == 0.01
    changed = copy.deepcopy(payload)
    changed["values"]["LW"]["7"]["emitted_phase_cuda"] = np.nan
    with pytest.raises(ValueError, match="reference TTC"):
        _validate_cuda_reference(changed, expected, record, "a" * 64)
