import pytest
import torch

from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.multimodal_research.phase_units import PhaseUnitAnchor
from operational.multimodal_research.relative_phase import RelativePhaseAnchor


@pytest.mark.parametrize("mode", ["EVENT", "ZERO"])
def test_relative_initialization_preserves_native_point_and_quantiles(mode):
    torch.manual_seed(23)
    values = torch.randn(12, 340)
    values[:, 160] = torch.tensor(
        [-1.0, -0.03, -0.001, -1e-8, 0.0, 1e-8, 0.001, 0.03, 0.1, 0.2, 1.0, 3.0]
    )
    values[:, 161] = emitted_phase(values[:, 160])
    widths = torch.randn(12, 2)
    baseline, relative = PhaseUnitAnchor(mode).eval(), RelativePhaseAnchor(mode).eval()
    relative.load_state_dict(baseline.state_dict())
    original = values.clone()
    with torch.no_grad():
        a, b = baseline(values, widths), relative(values, widths)
        assert all(torch.equal(a[key], b[key]) for key in a)
        assert torch.equal(values, original)
        if mode == "ZERO":
            relative.output.weight.normal_(std=0.1)
            perturbed = values.clone()
            perturbed[:, 170:335] = torch.randn(12, 165) * 1000
            a, b = relative(values, widths), relative(perturbed, widths)
            assert all(torch.equal(a[key], b[key]) for key in a)


def test_relative_correction_uses_predicted_magnitude_without_a_new_cap():
    model = RelativePhaseAnchor("EVENT")
    values = torch.zeros(4, 340)
    values[:, 160] = torch.tensor([-0.003, -0.03, 0.003, 0.03])
    values[:, 161] = emitted_phase(values[:, 160])
    with torch.no_grad():
        model.output.bias[0] = 2.0
    output = model(values, torch.zeros(4, 2))
    delta = output["raw_location"] - values[:, 160]
    torch.testing.assert_close(delta, 2 * values[:, 161].abs())
    assert delta[1] > 0.03
    output["raw_location"].sum().backward()
    torch.testing.assert_close(model.output.bias.grad[0], values[:, 161].abs().sum())
