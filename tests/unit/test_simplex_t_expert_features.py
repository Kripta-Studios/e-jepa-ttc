"""Reject incoherent extraction inputs before invoking any expert."""

from unittest.mock import Mock

import pytest
import torch

from e_jepa_ttc.simplex_t.expert_features import extract_family


@pytest.mark.parametrize("case", ["precision", "windows", "delta", "training"])
def test_extraction_rejects_invalid_contract_before_inference(case):
    models = [Mock(training=False) for _ in range(3)]
    events = torch.zeros(1, 3, 12, 2, 2)
    delta = torch.full((1, 2), 0.1)
    if case == "precision":
        events = events.double()
    elif case == "windows":
        events = events[:, :2]
    elif case == "delta":
        delta.zero_()
    else:
        models[0].training = True
    with pytest.raises(ValueError):
        extract_family(*models, events, delta)
    assert all(not model.called for model in models)
