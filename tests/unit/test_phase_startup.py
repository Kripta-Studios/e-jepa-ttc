import numpy as np
import torch

from operational.phase_robust.geometry import expansion_features
from operational.phase_startup.geometry import startup_geometry
from operational.phase_startup.model import ExpertRouter


def test_two_observations_resolve_current_ttc_but_one_cannot():
    times = np.array([0.0, 0.04, 0.08, 0.12])
    size = 100 / (2 - times)
    boxes = np.column_stack((-size / 2, -size / 2, size / 2, size / 2))
    clocks = np.rint(times * 1e6).astype(np.int64)
    single = startup_geometry(boxes, clocks, 0)
    assert not single["valid"] and single["quality"][9] == 1 / 32
    pair = startup_geometry(boxes, clocks, 40000)
    assert pair["valid"] and pair["count"] == 2
    np.testing.assert_allclose(pair["ttc"], 1.96, atol=1e-10)
    full = startup_geometry(boxes, clocks, 120000)
    old = expansion_features(boxes, clocks, 120000)
    assert full == old
    boxes[-1] = [0, 0, 10000, 10000]
    assert startup_geometry(boxes, clocks, 40000) == pair


def test_router_can_override_wrong_native_sign_and_is_sign_equivariant():
    model = ExpertRouter("quality")
    with torch.no_grad():
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 10.0]))
    native = torch.tensor([[-42.0, -26.0, 7.0]])
    geometry = torch.tensor([2.0])
    quality = torch.ones(1, 20)
    assert model(native, geometry, quality)[0] > 1.99
    reverse = quality.clone()
    reverse[:, :5] *= -1
    torch.testing.assert_close(
        model(-native, -geometry, reverse), -model(native, geometry, quality)
    )


def test_router_masks_missing_experts_and_preserves_no_history_fallback_exactly():
    model = ExpertRouter("quality")
    native = torch.tensor([[1.234567, 2.0, 3.0], [float("nan"), float("nan"), float("nan")]])
    geo = torch.tensor([float("nan"), 2.0])
    q = torch.ones(2, 20)
    q[0] = 0
    prediction = model(native, geo, q)
    assert torch.equal(prediction[:1], native[:1, 0])
    assert prediction[1] == 2
    assert torch.isnan(model(native[1:], torch.tensor([float("nan")]), q[1:])[0])
    assert sum(p.numel() for p in model.parameters()) == 688
    assert sum(p.numel() for p in ExpertRouter("native").parameters()) == 688
    scripted = torch.jit.script(model)
    torch.testing.assert_close(prediction, scripted(native, geo, q))


def test_inverse_coordinate_mixes_expansion_rates_and_preserves_sign_symmetry():
    model = ExpertRouter("quality", coordinate="inverse")
    with torch.no_grad():
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.zero_()
    native = torch.tensor([[1.0, 4.0, float("nan")]])
    quality = torch.ones(1, 20)
    geo = torch.tensor([float("nan")])
    torch.testing.assert_close(model(native, geo, quality), torch.tensor([1.6]))
    reverse = quality.clone()
    reverse[:, :5] *= -1
    torch.testing.assert_close(model(-native, geo, reverse), torch.tensor([-1.6]))
    script = torch.jit.script(model)
    torch.testing.assert_close(script(native, geo, quality), model(native, geo, quality))
    cancellation = torch.tensor([[2.0, -2.0, float("nan")]])
    torch.testing.assert_close(model(cancellation, geo, quality), cancellation[:, 0])
    torch.testing.assert_close(model(-cancellation, geo, reverse), -cancellation[:, 0])
