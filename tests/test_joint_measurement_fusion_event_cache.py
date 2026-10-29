"""Event feature admission must not bypass the C1 interpretation boundary."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.joint_measurement import fusion_event_cache as event
from operational.joint_measurement.fusion_pair_extract import extract_pair
from operational.joint_measurement.geometry_model import GeometricPairMeasurement


@pytest.mark.parametrize("fit,rationale", [("NATIVE-P2-FT-S7", "candidate"), (event.FIT, " ")])
def test_missing_or_unsupported_choice_cannot_read_artifacts(monkeypatch, fit, rationale):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject choice before artifact access")

    monkeypatch.setattr(event, "read", forbidden)
    with pytest.raises(ValueError, match="Declare"):
        event.register(fit, rationale)


def test_c1_training_or_partial_evidence_cannot_admit_event_parent(monkeypatch):
    monkeypatch.setattr(event, "read", lambda path: {"status": "EVALUATING"})

    def forbidden():
        raise AssertionError("Cannot inspect endpoints before complete C1 evidence")

    monkeypatch.setattr(event, "freeze_endpoint", forbidden)
    with pytest.raises(ValueError, match="Complete and interpret"):
        event.register(event.FIT, "Pending C1 interpretation")


def test_geometry_descriptor_extraction_matches_actual_native_model():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    parent = CausalScaleTTC(CausalScaleTTCConfig())
    model = GeometricPairMeasurement(parent).eval().requires_grad_(False)
    inputs = torch.randn(2, 2, 12, 128, 128)
    dt = torch.tensor([0.05, 0.100002])
    original = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    features, predictions = extract_pair(model, inputs, dt)
    with torch.inference_mode():
        output = model(inputs, dt)
    np.testing.assert_array_equal(features[:, :194], output["descriptor"].numpy())
    np.testing.assert_array_equal(predictions, phase_to_ttc(output["raw_phase"]).numpy())
    assert features.shape == (2, 196)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
