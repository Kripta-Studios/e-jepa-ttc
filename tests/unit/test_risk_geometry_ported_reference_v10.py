"""Reference cases ported to production imports; lifecycle covered independently."""
# ruff: noqa: E741, N806 -- retain frozen reference mathematical variable notation

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.models.normal_flow_routing_v10 import (
    fit_affine_normal_flow,
    fit_time_plane,
    inverse_ttc_at_anchor,
)
from e_jepa_ttc.models.simplex_risk_router import (
    SimplexRiskRouter,
    benchmark_phase,
    costs_from_logits,
    hard_select,
    relative_cost_targets,
    weighted_cost_loss,
)

torch.set_num_threads(1)


@pytest.mark.parametrize("bad", [0.0, 0.1, 0.05, float("nan"), float("inf")])
def test_phase_invalid(bad):
    with pytest.raises(ValueError):
        benchmark_phase(np.array([bad]))


def test_phase_signed_and_roundtrip():
    t = np.array([-10.0, -1.0, -0.1, 0.101, 1.0, 10.0, 1e6])
    p = benchmark_phase(t)
    np.testing.assert_allclose(0.1 / (-np.expm1(-p)), t, rtol=1e-11)
    assert np.array_equal(np.sign(t), np.sign(p))


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_absolute_risk_structure(seed):
    g = torch.Generator().manual_seed(seed)
    phases = torch.randn(100, 3, generator=g, dtype=torch.float64) * 0.1
    logits = torch.randn(100, 2, generator=g, dtype=torch.float64) * 5
    costs = costs_from_logits(phases, logits)
    order = torch.argsort(phases, dim=1, stable=True)
    ps = phases.gather(1, order)
    cs = costs.gather(1, order)
    slopes = (cs[:, 1:] - cs[:, :-1]) / (100 * (ps[:, 1:] - ps[:, :-1]))
    assert (slopes >= -1 - 1e-12).all() and (slopes <= 1 + 1e-12).all()
    assert (slopes[:, 1] >= slopes[:, 0] - 1e-12).all()
    for i, j in itertools.combinations(range(3), 2):
        assert (
            (costs[:, i] - costs[:, j]).abs() <= 100 * (phases[:, i] - phases[:, j]).abs() + 1e-12
        ).all()


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
def test_expert_permutation_equivariance(permutation):
    p = torch.tensor([[-0.04, 0.08, 0.02]], dtype=torch.float64)
    l = torch.tensor([[0.2, -0.7]], dtype=torch.float64)
    cost = costs_from_logits(p, l)
    perm = torch.tensor(permutation)
    changed = costs_from_logits(p[:, perm], l)
    expected = cost[:, perm] - cost[:, perm][:, :1]
    torch.testing.assert_close(changed, expected, atol=1e-14, rtol=0)


def test_duplicate_expert_phase_equal_cost():
    p = torch.tensor([[0.02, 0.02, 0.1], [-0.03, 0.04, 0.04], [0.1, 0.1, 0.1]], dtype=torch.float64)
    c = costs_from_logits(
        p, torch.tensor([[4.0, -4.0], [4.0, -4.0], [4.0, -4.0]], dtype=torch.float64)
    )
    assert (
        c[0, 0] == c[0, 1]
        and c[1, 1] == c[1, 2]
        and torch.equal(c[2], torch.zeros(3, dtype=torch.float64))
    )


def test_zero_head_matched_initial_costs():
    p = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64)
    l = torch.zeros(1, 2, dtype=torch.float64)
    torch.testing.assert_close(
        costs_from_logits(p, l), costs_from_logits(p, l, constrained=False), atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("dim,expected", [(17, 1138), (45, 2034)])
def test_parameters_and_gradients(dim, expected):
    torch.manual_seed(7)
    m = SimplexRiskRouter(dim).double()
    assert sum(x.numel() for x in m.parameters()) == expected
    x = torch.randn(64, dim, dtype=torch.float64)
    p = torch.randn(64, 3, dtype=torch.float64) * 0.05
    y = torch.randn(64, dtype=torch.float64) * 0.05
    w = torch.full((64,), 1 / 64, dtype=torch.float64)
    loss = weighted_cost_loss(m(x, p), relative_cost_targets(y, p), w, 64)
    loss.backward()
    assert torch.isfinite(loss) and all(
        q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters()
    )
    assert m.head.weight.grad.abs().sum() > 0


def test_global_mass_unbiased_minibatch():
    pred = torch.zeros(4, 3, dtype=torch.float64)
    target = torch.tensor(
        [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0], [0.0, 4.0, 8.0]], dtype=torch.float64
    )
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    full = weighted_cost_loss(pred, target, w, 4)
    partial = torch.stack(
        [weighted_cost_loss(pred[i : i + 1], target[i : i + 1], w[i : i + 1], 4) for i in range(4)]
    ).mean()
    torch.testing.assert_close(full, partial)


def test_hard_selection_preserves_exact_signed_expert():
    costs = torch.tensor([[0.0, 0.0, 0.0], [0.0, -1.0, 2.0]])
    t = torch.tensor([[-2.0, 4.0, 6.0], [5.0, -3.0, 9.0]])
    idx, pred = hard_select(costs, t)
    assert idx.tolist() == [0, 1] and pred.tolist() == [-2.0, -3.0]


def test_simplex_synthetic_optimization():
    torch.manual_seed(7)
    m = SimplexRiskRouter().double()
    x = torch.randn(128, 17, dtype=torch.float64)
    phases = torch.tensor([[-0.04, 0.02, 0.08]], dtype=torch.float64).expand(128, -1)
    target = relative_cost_targets(torch.full((128,), 0.08, dtype=torch.float64), phases)
    w = torch.full((128,), 1 / 128, dtype=torch.float64)
    opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.01)
    first = float(weighted_cost_loss(m(x, phases), target, w, 128).detach())
    for _ in range(80):
        opt.zero_grad()
        loss = weighted_cost_loss(m(x, phases), target, w, 128)
        loss.backward()
        opt.step()
    last = float(weighted_cost_loss(m(x, phases), target, w, 128).detach())
    assert last < first * 0.05


def test_time_plane_reconstructs_normal_constraint():
    a, b = 0.4, -0.2
    xx, yy = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(-0.1, 0.1, 5))
    xy = np.c_[xx.ravel(), yy.ravel()]
    t = a * xy[:, 0] + b * xy[:, 1] + 2
    r = fit_time_plane(xy, t)
    assert r.valid and r.r_squared > 0.999999
    np.testing.assert_allclose(r.normal, [a, b] / np.linalg.norm([a, b]), atol=1e-12)
    assert r.speed == pytest.approx(1 / np.linalg.norm([a, b]))


@pytest.mark.parametrize("kind", ["sparse", "collinear", "constant"])
def test_bad_planes_invalid(kind):
    xy = np.c_[np.arange(16), np.arange(16) ** 2].astype(float)
    t = np.arange(16, dtype=float)
    if kind == "sparse":
        xy, t = xy[:4], t[:4]
    if kind == "collinear":
        xy[:, 1] = xy[:, 0]
    if kind == "constant":
        t[:] = 1
    assert not fit_time_plane(xy, t).valid


def fixture_affine(seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, (1024, 2))
    theta = rng.uniform(0, 2 * np.pi, 1024)
    normals = np.c_[np.cos(theta), np.sin(theta)]
    A = np.array([[0.4, 0.13], [-0.07, 0.6]])
    b = np.array([0.2, -0.1])
    v = xy @ A.T + b
    s = np.sum(normals * v, axis=1)
    return xy, normals, s, np.r_[A.ravel(), b]


def test_affine_exact_with_translation_rotation_shear():
    xy, n, s, beta = fixture_affine()
    fit = fit_affine_normal_flow(xy, n, s)
    assert fit.valid
    np.testing.assert_allclose(fit.beta, beta, atol=1e-12)
    assert fit.kappa == pytest.approx(0.5)


def test_affine_robust_outliers():
    xy, n, s, beta = fixture_affine()
    s = s.copy()
    s[:100] += 5
    fit = fit_affine_normal_flow(xy, n, s)
    assert fit.valid
    np.testing.assert_allclose(fit.beta, beta, atol=0.005)


def test_aperture_problem_is_not_false_valid():
    xy, n, s, _ = fixture_affine()
    n[:] = [1.0, 0.0]
    assert not fit_affine_normal_flow(xy, n, s).valid


def test_affine_coordinate_rescaling_trace_invariant():
    xy, n, s, beta = fixture_affine()
    D = np.array([2.0, 0.5])
    nn = n / D
    norm = np.linalg.norm(nn, axis=1)
    fit = fit_affine_normal_flow(xy * D, nn / norm[:, None], s / norm)
    assert fit.valid and fit.kappa == pytest.approx(0.5)


def test_ttc_anchor_correct_direction():
    assert inverse_ttc_at_anchor(0.5, 0.1) == pytest.approx(1 / 2.1)
    assert inverse_ttc_at_anchor(-0.5, 0.1) == pytest.approx(1 / -1.9)
    with pytest.raises(ValueError):
        inverse_ttc_at_anchor(0.5, -2)


def test_exact_bootstrap_enumeration_nine():
    r = exact_sequence_diagnostic(np.arange(-9.0, 0.0))
    assert r["sequence_count_vectors"] == 24310 and r["fraction_negative"] == pytest.approx(1.0)
    assert r["sequence_wins"] == 9 and not r["retraining_performed"]


def test_exact_bootstrap_matches_ordered_small():
    x = np.array([-2.0, 1.0, 4.0])
    r = exact_sequence_diagnostic(x)
    v = np.mean(np.array(list(itertools.product(x, repeat=3))), axis=1)
    assert r["fraction_negative"] == pytest.approx(float(np.mean(v < 0)))
    assert r["ci95"] == [
        float(np.quantile(v, 0.025, method="inverted_cdf")),
        float(np.quantile(v, 0.975, method="inverted_cdf")),
    ]


def test_optimizer_exact_ten_vs_five_plus_five():
    import copy

    torch.manual_seed(7)
    x = torch.randn(32, 17, dtype=torch.float64)
    p = torch.randn(32, 3, dtype=torch.float64) * 0.1
    target = relative_cost_targets(torch.randn(32, dtype=torch.float64) * 0.1, p)
    w = torch.full((32,), 1 / 32, dtype=torch.float64)
    torch.manual_seed(13)
    m = SimplexRiskRouter().double()
    o = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01)
    saved = None
    for step in range(10):
        o.zero_grad()
        weighted_cost_loss(m(x, p), target, w, 32).backward()
        o.step()
        if step == 4:
            saved = copy.deepcopy((m.state_dict(), o.state_dict(), torch.get_rng_state()))
    r = SimplexRiskRouter().double()
    ro = torch.optim.AdamW(r.parameters(), lr=0.001, weight_decay=0.01)
    r.load_state_dict(saved[0])
    ro.load_state_dict(saved[1])
    torch.set_rng_state(saved[2])
    for _ in range(5):
        ro.zero_grad()
        weighted_cost_loss(r(x, p), target, w, 32).backward()
        ro.step()
    for a, b in zip(m.parameters(), r.parameters(), strict=True):
        assert torch.equal(a, b)


def test_circle_expansion_identifiable_despite_unobservable_rotation():
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    n = np.c_[np.cos(theta), np.sin(theta)]
    xy = 0.3 * n
    s = 0.6 * np.sum(n * xy, axis=1) + n @ np.array([0.2, -0.1])
    fit = fit_affine_normal_flow(xy, n, s)
    assert fit.valid and fit.mode == "isotropic" and fit.kappa == pytest.approx(0.6)
    np.testing.assert_allclose(fit.beta[4:], [0.2, -0.1], atol=1e-12)


def test_pure_translation_not_reported_as_collision_expansion():
    xy, n, _, _ = fixture_affine()
    s = n @ np.array([0.8, -0.4])
    fit = fit_affine_normal_flow(xy, n, s)
    assert fit.valid and abs(fit.kappa) < 1e-12


@pytest.mark.parametrize("invalid", [-0.05, 0.1])
def test_hard_selection_strict_prediction_domain(invalid):
    costs = torch.zeros(1, 3, dtype=torch.float64)
    ttc = torch.tensor([[invalid, 2.0, -2.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        hard_select(costs, ttc)
