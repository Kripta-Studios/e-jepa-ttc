import pytest
import torch

from e_jepa_ttc.simplex_t.controls import ewma_phase, history_control
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
from e_jepa_ttc.simplex_t.phase import (
    emitted_phase,
    phase_to_ttc,
    relative_cost_targets,
    ttc_to_phase,
)


def batch(n=3, length=8, f=17):
    x = torch.randn(n, length, f)
    t = torch.zeros(n, length, 4)
    t[:, :, 0] = torch.arange(length - 1, -1, -1) * 0.1
    v = torch.ones(n, length, dtype=torch.bool)
    v[0, : length // 2] = False
    e = torch.tensor([[0.02, 0.03, 0.04]]).repeat(n, 1)
    return x, t, v, e


@pytest.mark.parametrize("ttc", [-60.0, -1.0, -0.101, 0.101, 1.0, 60.0])
def test_phase_roundtrip(ttc):
    value = torch.tensor([ttc], dtype=torch.float64)
    torch.testing.assert_close(phase_to_ttc(ttc_to_phase(value)), value, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, 0.1, 0.05, float("nan")])
def test_bad_ttc(bad):
    with pytest.raises(ValueError):
        ttc_to_phase(torch.tensor([bad]))


def test_zero_maps_positive60():
    assert phase_to_ttc(torch.tensor([0.0])).item() == pytest.approx(60)


@pytest.mark.parametrize("backbone,h", [("gru", 64), ("gru", 160), ("transformer", 128)])
def test_initial_anchor_and_padding(backbone, h):
    model = TemporalRefiner(TemporalConfig(hidden=h, backbone=backbone)).eval()
    x, t, v, e = batch()
    out = model(x, t, v, e)
    torch.testing.assert_close(out["point_phase"], emitted_phase(e.median(-1).values))
    assert (out["q10"] <= out["raw_location"]).all() and (out["q90"] >= out["raw_location"]).all()
    x[~v] = float("nan")
    t[~v] = float("nan")
    out2 = model(x, t, v, e)
    torch.testing.assert_close(out["point_phase"], out2["point_phase"])


@pytest.mark.parametrize("backbone,h", [("gru", 64), ("transformer", 128)])
def test_padding_invariance_nonzero_head(backbone, h):
    model = TemporalRefiner(TemporalConfig(hidden=h, backbone=backbone)).eval()
    torch.nn.init.normal_(model.location.weight, std=0.01)
    x, t, v, e = batch(n=1, length=4)
    v[:] = True
    a = model(x, t, v, e)["raw_location"]
    xp = torch.cat([torch.randn(1, 4, 17), x], 1)
    tp = torch.cat([torch.randn(1, 4, 4), t], 1)
    vp = torch.cat([torch.zeros(1, 4, dtype=torch.bool), v], 1)
    b = model(xp, tp, vp, e)["raw_location"]
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("bias", [-10.0, 10.0])
def test_escape_expert_hull(bias):
    model = TemporalRefiner(TemporalConfig())
    with torch.no_grad():
        model.location.bias.fill_(bias)
    x, t, v, e = batch()
    p = model(x, t, v, e)["point_phase"]
    assert ((p < e.min(-1).values) if bias < 0 else (p > e.max(-1).values)).all()


def test_quantiles_order_after_large_shift():
    model = TemporalRefiner(TemporalConfig())
    with torch.no_grad():
        model.location.bias.fill_(1000)
    q = model(*batch())
    assert (q["q10"] <= q["q90"]).all()


def test_reject_empty_current():
    x, t, v, e = batch()
    v[:, -1] = False
    with pytest.raises(ValueError):
        TemporalRefiner(TemporalConfig())(x, t, v, e)


def test_reject_nonfinite_valid():
    x, t, v, e = batch()
    x[:, -1, 0] = float("nan")
    with pytest.raises(ValueError):
        TemporalRefiner(TemporalConfig())(x, t, v, e)


def test_cost_target_units():
    e = torch.tensor([[0.01, 0.03, 0.10]])
    y = torch.tensor([0.02])
    torch.testing.assert_close(
        relative_cost_targets(e, y), torch.tensor([[0.0, 7.0]]), atol=1e-6, rtol=1e-6
    )


def test_global_weight_no_batch_normalize():
    model = TemporalRefiner(TemporalConfig())
    x, t, v, e = batch(4)
    y = torch.tensor([0.01, 0.03, 0.07, -0.03])
    mass = torch.tensor([0.1, 0.1, 0.2, 0.6])
    o = model(x, t, v, e)
    all_loss = training_loss(o, y, e, mass, 4)
    parts = []
    for sl in (slice(0, 2), slice(2, 4)):
        oo = {k: z[sl] for k, z in o.items()}
        parts.append(training_loss(oo, y[sl], e[sl], mass[sl], 4))
    torch.testing.assert_close(all_loss, sum(parts) / 2)


def test_gradients_and_selector():
    model = TemporalRefiner(TemporalConfig())
    x, t, v, e = batch()
    y = torch.tensor([0.01, 0.08, -0.03])
    loss = training_loss(model(x, t, v, e), y, e, torch.ones(3) / 3, 3)
    loss.backward()
    assert model.location.weight.grad.abs().sum() > 0 and model.cost.weight.grad.abs().sum() > 0
    selector = TemporalRefiner(TemporalConfig(output_mode="selector"))
    out = selector(x, t, v, e)
    torch.testing.assert_close(out["point_phase"], e[:, 0])


def test_controls_hold_current_and_mask():
    x, t, v, e = batch()
    for kind in ("repeat", "shuffle"):
        changed = history_control(x, v, kind)
        torch.testing.assert_close(changed[:, -1], x[:, -1])
        torch.testing.assert_close(changed[~v], x[~v])


def test_ewma_constant():
    e = torch.full((3, 8, 3), 0.02)
    lags = torch.arange(7, -1, -1)[None].repeat(3, 1) * 0.1
    p = ewma_phase(e, lags, torch.ones(3, 8, dtype=torch.bool))
    torch.testing.assert_close(p, emitted_phase(torch.full((3,), 0.02)))


def test_latent_input():
    model = TemporalRefiner(TemporalConfig(feature_count=145, hidden=160))
    p = model(*batch(f=145))
    assert p["point_phase"].shape == (3,)


def test_free_head_not_anchor():
    model = TemporalRefiner(TemporalConfig(output_mode="free"))
    p = model(*batch())
    torch.testing.assert_close(p["raw_location"], torch.zeros(3))


def test_gradient_accumulation_matches_whole_batch():
    torch.manual_seed(2)
    model = TemporalRefiner(TemporalConfig())
    x, t, v, e = batch(4)
    y = torch.tensor([0.01, 0.03, 0.07, -0.03])
    mass = torch.tensor([0.1, 0.1, 0.2, 0.6])
    full = training_loss(model(x, t, v, e), y, e, mass, 4)
    full.backward()
    full_grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for sl in (slice(0, 2), slice(2, 4)):
        loss = training_loss(model(x[sl], t[sl], v[sl], e[sl]), y[sl], e[sl], mass[sl], 4)
        (loss / 2).backward()
    for k, p in model.named_parameters():
        if k in full_grads:
            torch.testing.assert_close(full_grads[k], p.grad, atol=1e-6, rtol=1e-5)
