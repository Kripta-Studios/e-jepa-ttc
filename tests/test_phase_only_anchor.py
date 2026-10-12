"""Capacity and modality controls for the phase/clock-only corrector."""

import torch

from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.multimodal_research.phase_only_anchor import PhaseOnlyAnchor


def test_hidden_states_cannot_change_correction_inputs() -> None:
    torch.manual_seed(321)
    model = PhaseOnlyAnchor("EVENT")
    features = torch.randn(7, 340)
    changed = features.clone()
    changed[:, :160] *= 100
    changed[:, 170:330] += 100
    torch.testing.assert_close(model.inputs(features), model.inputs(changed), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 1955


def test_control_preserves_clocks_and_initial_rgb_anchor() -> None:
    features = torch.randn(7, 340)
    full, control = PhaseOnlyAnchor("EVENT"), PhaseOnlyAnchor("ZERO")
    a, b = full.inputs(features), control.inputs(features)
    torch.testing.assert_close(a[:, :10], b[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(a[:, 15:], b[:, 15:], rtol=0, atol=0)
    assert not torch.count_nonzero(b[:, 10:15])
    for model in (full, control):
        prediction = model(features, torch.randn(7, 2))["point_phase"]
        torch.testing.assert_close(prediction, emitted_phase(features[:, 160]), rtol=0, atol=0)
