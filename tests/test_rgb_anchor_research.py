import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.multimodal_research.model import BLOCK, RGBAnchor, frozen_state


def test_anchor_preserves_real_refiner_and_quantiles():
    torch.manual_seed(19)
    model = TemporalRefiner(TemporalConfig(hidden=160)).eval()
    with torch.no_grad():
        model.location.weight.normal_(std=0.1)
        model.width.weight.normal_(std=0.1)
    x, clock = torch.randn(7, 8, 17), torch.randn(7, 8, 4)
    valid = torch.ones(7, 8, dtype=torch.bool)
    valid[0, :5] = False
    expert = torch.randn(7, 3) * 0.03
    with torch.no_grad():
        original = model(x, clock, valid, expert)
        block, width, observed = frozen_state(model, x, clock, valid, expert)
        for key in original:
            assert torch.equal(original[key], observed[key])
        for mode in ("EVENT", "ZERO"):
            prediction = RGBAnchor(mode)(torch.cat([block, block], -1), width)
            for key in ("raw_location", "point_phase", "q10", "q90"):
                assert torch.equal(prediction[key], original[key])
    assert not model.location._forward_pre_hooks
    assert not model.width._forward_hooks


def test_zero_control_removes_every_predictive_event_coordinate_after_learning():
    torch.manual_seed(23)
    model = RGBAnchor("ZERO").eval()
    with torch.no_grad():
        model.output.weight.normal_(std=0.1)
        model.output.bias.normal_(std=0.1)
        x = torch.randn(8, 2 * BLOCK)
        altered = x.clone()
        altered[:, BLOCK : BLOCK + 165] = torch.randn(8, 165) * 1000
        width = torch.randn(8, 2)
        a, b = model(x, width), model(altered, width)
        for key in a:
            assert torch.equal(a[key], b[key])
        event = RGBAnchor("EVENT").eval()
        event.load_state_dict(model.state_dict(), strict=True)
        assert not torch.equal(
            event(x, width)["raw_location"], event(altered, width)["raw_location"]
        )
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in event.parameters()
        )
