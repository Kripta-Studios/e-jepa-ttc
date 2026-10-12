"""Preserve F_TRUE at initialization and isolate only additional refined RGB evidence."""

import torch
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase
from operational.multimodal_research.fusion_preserving_anchor import FusionPreservingAnchor


def test_initialization_preserves_fusion_point_and_widths() -> None:
    torch.manual_seed(103)
    features = torch.randn(9, 502)
    state = {"width.weight": torch.randn(2, 160), "width.bias": torch.randn(2)}
    for mode in ("REFINED_RGB", "ZERO_RGB"):
        model = FusionPreservingAnchor(mode)
        model.initialize_width(state)
        actual = model(features, torch.zeros(9, 2))
        raw = features[:, 500]
        widths = 0.03 * functional.softplus(
            functional.linear(features[:, 340:500], state["width.weight"], state["width.bias"])
        )
        torch.testing.assert_close(actual["point_phase"], emitted_phase(raw), rtol=0, atol=0)
        torch.testing.assert_close(
            actual["q10"], (raw - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["q90"], (raw + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE), rtol=0, atol=0
        )
        assert not model.frozen_width_weight.requires_grad


def test_control_removes_only_refined_rgb_predictive_coordinates() -> None:
    features = torch.randn(8, 502)
    full = FusionPreservingAnchor("REFINED_RGB").inputs(features)
    control = FusionPreservingAnchor("ZERO_RGB").inputs(features)
    torch.testing.assert_close(full[:, 165:], control[:, 165:], rtol=0, atol=0)
    assert not torch.count_nonzero(control[:, :165])
    assert torch.count_nonzero(control[:, 340:]) > 0


def test_residual_is_trainable_without_restricting_it_to_expert_interval() -> None:
    model = FusionPreservingAnchor("REFINED_RGB")
    features = torch.zeros(2, 502)
    features[:, 500] = 0.02
    with torch.no_grad():
        model.output.bias[0] = 2
    result = model(features, torch.zeros(2, 2))
    torch.testing.assert_close(result["raw_location"], torch.full((2,), 0.08))
    result["point_phase"].sum().backward()
    assert model.output.bias.grad is not None and model.output.bias.grad[0] != 0
