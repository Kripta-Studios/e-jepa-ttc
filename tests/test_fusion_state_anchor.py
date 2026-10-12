import pytest
import torch

from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.multimodal_research.fusion_anchor import FusionStateAnchor
from operational.multimodal_research.phase_units import PhaseUnitAnchor


@pytest.mark.parametrize("mode", ["FUSED", "ZERO_FUSED"])
def test_fusion_state_initial_point_and_quantiles_equal_rgb_anchor(mode):
    torch.manual_seed(37)
    x = torch.randn(12, 502)
    x[:, 160] = torch.tensor(
        [-1.0, -0.03, -0.001, -1e-8, 0.0, 1e-8, 0.001, 0.03, 0.1, 0.2, 1.0, 3.0]
    )
    x[:, 161] = emitted_phase(x[:, 160])
    original = x.clone()
    widths = torch.randn(12, 2)
    base, augmented = PhaseUnitAnchor("EVENT"), FusionStateAnchor(mode)
    with torch.no_grad():
        a, b = base(x[:, :340], widths), augmented(x, widths)
        assert all(torch.equal(a[key], b[key]) for key in a)
        assert torch.equal(x, original)


def test_zero_fused_nulls_only_added_fusion_information():
    torch.manual_seed(29)
    model = FusionStateAnchor("ZERO_FUSED")
    x, widths = torch.randn(12, 502), torch.randn(12, 2)
    altered = x.clone()
    altered[:, 340:] = torch.randn(12, 162) * 1000
    with torch.no_grad():
        model.output.weight.normal_(std=0.1)
        a, b = model(x, widths), model(altered, widths)
        assert all(torch.equal(a[key], b[key]) for key in a)
        changed_event = x.clone()
        changed_event[:, 170:335] = torch.randn(12, 165) * 100
        assert not torch.equal(model.inputs(x), model.inputs(changed_event))
    reference = FusionStateAnchor("FUSED")
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
