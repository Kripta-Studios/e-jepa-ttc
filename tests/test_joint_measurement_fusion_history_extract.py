"""Check native RGB-anchor parity and reject incompatible feature histories."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.joint_measurement.fusion_history_extract import FEATURE_NAMES, extract_history


def fixture():
    torch.manual_seed(71)
    model = TemporalRefiner(TemporalConfig(hidden=160)).eval().requires_grad_(False)
    with torch.no_grad():
        model.location.weight.normal_(0, 0.001)
    features = torch.randn(3, 8, 17)
    timing = torch.randn(3, 8, 4)
    valid = torch.ones(3, 8, dtype=torch.bool)
    valid[0, :6] = False
    features[0, :6] = torch.nan
    timing[0, :6] = torch.nan
    experts = torch.tensor([[-0.2, -0.1, -0.15], [0.1, 0.2, 0.3], [0.02, 0.04, 0.03]])
    return model, features, timing, valid, experts


def test_history_matches_native_anchor_without_changing_state():
    model, features, timing, valid, experts = fixture()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        reference = model(features, timing, valid, experts)
    descriptors, predictions = extract_history(model, features, timing, valid, experts)
    assert descriptors.shape == (3, len(FEATURE_NAMES)) == (3, 28)
    np.testing.assert_array_equal(predictions, phase_to_ttc(reference["point_phase"]).numpy())
    np.testing.assert_array_equal(descriptors[:, :17], features[:, -1].numpy())
    np.testing.assert_array_equal(descriptors[:, 17:21], timing[:, -1].numpy())
    np.testing.assert_array_equal(descriptors[:, 21:24], experts.numpy())
    assert predictions[0] < 0 and (predictions[1:] > 0).all()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("case", ["train", "grad", "width", "dtype", "hole", "missing"])
def test_rejects_unadmitted_history_inputs(case):
    model, features, timing, valid, experts = fixture()
    if case == "train":
        model.train()
    elif case == "grad":
        model.location.weight.requires_grad_(True)
    elif case == "width":
        model = TemporalRefiner(TemporalConfig(hidden=64)).eval().requires_grad_(False)
    elif case == "dtype":
        features = features.double()
    elif case == "hole":
        valid[1, 4] = False
    else:
        valid[1, -1] = False
    with pytest.raises(ValueError):
        extract_history(model, features, timing, valid, experts)
