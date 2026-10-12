import pytest
import torch

from operational.multimodal_research.model import RGBAnchor
from operational.multimodal_research.phase_units import PhaseUnitAnchor


@pytest.mark.parametrize("mode", ["EVENT", "ZERO"])
def test_units_preserve_original_anchor_and_do_not_mutate_cached_states(mode):
    torch.manual_seed(41)
    features = torch.randn(12, 340)
    features[:, 160] = torch.tensor(
        [-1.0, -0.03, -0.001, -1e-8, 0.0, 1e-8, 0.001, 0.03, 0.1, 0.2, 1.0, 3.0]
    )
    original = features.clone()
    widths = torch.randn(12, 2)
    base, units = RGBAnchor(mode).eval(), PhaseUnitAnchor(mode).eval()
    units.load_state_dict(base.state_dict())
    with torch.no_grad():
        a, b = base(features, widths), units(features, widths)
        for key in a:
            assert torch.equal(a[key], b[key])
        assert torch.equal(features, original)
        if mode == "ZERO":
            units.output.weight.normal_(std=0.1)
            altered = features.clone()
            altered[:, 170:335] *= 1000
            a, b = units(features, widths), units(altered, widths)
            for key in a:
                assert torch.equal(a[key], b[key])
