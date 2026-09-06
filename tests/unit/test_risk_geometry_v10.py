"""Integrated integrity, controls, geometry, and complete resume regression tests."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_decision_v10 import DecisionBlocked, stage_decision
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    ORDER,
    AllowlistedSources,
    PhaseLedger,
    binding,
    object_digest,
)
from e_jepa_ttc.data.event_geometry_cache_v10 import combine_windows, geometry_window
from e_jepa_ttc.data.frozen_expert_tables_v10 import (
    ModelInputs,
    arm_inputs,
    feature_mask,
    load_table,
    pair_permutation,
)
from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.models.normal_flow_routing_v10 import fit_affine_normal_flow
from e_jepa_ttc.models.simplex_risk_router import SimplexRiskRouter, costs_from_logits
from e_jepa_ttc.training.risk_router_v10 import scaler, train_steps

torch.set_num_threads(1)


@pytest.mark.parametrize("state", ORDER[2:])
def test_cannot_skip_any_prerequisite(tmp_path, state):
    p = tmp_path / "receipt.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        PhaseLedger(tmp_path / "ledger.json").advance(state, [p])


@pytest.mark.parametrize("count", range(1, len(ORDER)))
def test_deleted_or_modified_receipt_blocks_all_successors(tmp_path, count):
    ledger = PhaseLedger(tmp_path / "ledger.json")
    for i, state in enumerate(ORDER[1 : count + 1]):
        p = tmp_path / f"{i}.json"
        p.write_text("{}")
        ledger.advance(state, [p])
    (tmp_path / "0.json").write_text('{"forged":true}')
    with pytest.raises(ValueError):
        PhaseLedger(tmp_path / "ledger.json")


def test_exact_allowlist_not_path_substring(tmp_path):
    good = tmp_path / "train.json"
    bad = tmp_path / "public_validation.json"
    good.write_text("{}")
    bad.write_text("{}")
    access = AllowlistedSources([binding(good)], tmp_path / "journal.jsonl")
    assert access.access(good) == good.resolve()
    with pytest.raises(PermissionError):
        access.access(bad)
    good.write_text("changed")
    with pytest.raises(ValueError):
        access.access(good)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("dim", [17, 45])
def test_full_state_ten_versus_five_resume_five(tmp_path, real, dim):
    rng = np.random.default_rng(12)
    if real:
        index = Path("artifacts/stage66_69_clean_v2/TABLE_INDEX.json")
        if not index.exists():
            pytest.skip("explicit local inner-OOF smoke input unavailable")
        table = load_table(json.loads(index.read_text())["outer0_inner_oof"], role="inner_oof")
        x = table.inputs.features[:128].copy()
        p = table.inputs.phases[:128].copy()
        t = table.inputs.expert_ttc[:128].copy()
        y = table.supervision.target_phase[:128].copy()
    else:
        x = rng.normal(size=(128, 17))
        p = rng.uniform(0.01, 0.1, size=(128, 3))
        t = 0.1 / -np.expm1(-p)
        y = rng.uniform(0.01, 0.1, 128)
    if dim == 45:
        x = np.concatenate([x, rng.normal(size=(128, 28))], 1)
    mass = np.ones(128) / 128
    mean, std = scaler(x, mass)
    inputs = ModelInputs(x, p, t)
    kwargs = dict(
        inputs=inputs,
        target_phase=y,
        mass=mass,
        mean=mean,
        std=std,
        arm="S67-SIMPLEX17",
        seed=7,
        outer=0,
        identity={"purpose": "qa", "real_inner_oof": real},
        total_updates=10,
    )
    continuous = train_steps(**kwargs, output=tmp_path / "continuous")
    train_steps(**kwargs, output=tmp_path / "resumed", stop_at=5)
    resumed = train_steps(**kwargs, output=tmp_path / "resumed", resume=True)
    a = torch.load(continuous["path"], weights_only=False)
    b = torch.load(resumed["path"], weights_only=False)
    # Extra full-train measurement at interruption is observational, not optimizer state.
    a.pop("full_losses")
    b.pop("full_losses")
    a.pop("started_ns")
    b.pop("started_ns")
    assert_equal(a, b)


def test_resume_identity_cannot_change(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 17))
    p = np.full((32, 3), 0.1)
    m = np.ones(32) / 32
    kwargs = dict(
        inputs=ModelInputs(x, p, np.full_like(p, 2)),
        target_phase=np.ones(32) * 0.2,
        mass=m,
        mean=x.mean(0),
        std=x.std(0),
        arm="S67-SIMPLEX17",
        seed=7,
        outer=0,
        identity={"purpose": "qa"},
        total_updates=10,
        output=tmp_path / "fit",
    )
    train_steps(**kwargs, stop_at=5)
    with pytest.raises(ValueError):
        train_steps(**{**kwargs, "seed": 13}, resume=True)


def test_real_role_forgery_and_uncertainty_mutation_rejected():
    index = Path("artifacts/stage66_69_clean_v2/TABLE_INDEX.json")
    if not index.exists():
        pytest.skip("local smoke table unavailable")
    rec = json.loads(index.read_text())["outer0_outer_dev"]
    table = load_table(rec, role="outer_dev")
    with pytest.raises(ValueError):
        table.validate("inner_oof")
    forged = copy.deepcopy(rec)
    forged["role"] = "inner_oof"
    with pytest.raises(ValueError):
        replace(table, record=forged, seal=object_digest(forged)).validate("inner_oof")
    table.inputs.features[0, 4] += 1
    with pytest.raises(ValueError):
        table.validate("outer_dev")


def test_real_pair_permutation_consistency():
    index = Path("artifacts/stage66_69_clean_v2/TABLE_INDEX.json")
    if not index.exists():
        pytest.skip("local smoke table unavailable")
    table = load_table(json.loads(index.read_text())["outer0_inner_oof"], role="inner_oof")
    donor = pair_permutation(table, 70)
    changed = arm_inputs(table, "S67-SIMPLEX17-PAIRPERM", donor)
    assert np.all(table.metadata.track_id.to_numpy() != table.metadata.track_id.to_numpy()[donor])
    assert np.array_equal(changed.expert_ttc[:, 2], table.inputs.expert_ttc[donor, 2])
    np.testing.assert_array_equal(
        changed.features[:, 11], changed.phases[:, 2] - changed.phases[:, 0]
    )
    g = np.arange(len(donor) * 28).reshape(-1, 28).astype(float)
    shuffled = arm_inputs(table, "S69-GSHUFFLE45", donor, g)
    np.testing.assert_array_equal(shuffled.features[:, 17:], g[donor])


@pytest.mark.parametrize("dim,expected", [(17, 1138), (45, 2034)])
def test_parameters_gradient_geometry_and_control_masks(dim, expected):
    torch.manual_seed(7)
    m = SimplexRiskRouter(dim).double()
    assert sum(p.numel() for p in m.parameters()) == expected
    x = torch.randn(50, dim, dtype=torch.float64)
    phases = torch.randn(50, 3, dtype=torch.float64)
    result = m(x, phases)
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    order = torch.argsort(phases, dim=1, stable=True)
    p = phases.gather(1, order)
    c = result.gather(1, order)
    slopes = torch.diff(c, dim=1) / (100 * torch.diff(p, dim=1))
    assert (slopes[:, 0] <= slopes[:, 1] + 1e-12).all() and (slopes.abs() <= 1 + 1e-12).all()
    if dim == 45:
        assert feature_mask("S69-GQUALITY45", 45).sum() == 29


def test_duplicate_phases_and_initial_unconstrained_match():
    p = torch.tensor([[0.1, 0.1, 0.2], [0.2, 0.2, 0.2]], dtype=torch.float64)
    z = torch.zeros(2, 2, dtype=torch.float64)
    a = costs_from_logits(p, z)
    b = costs_from_logits(p, z, constrained=False)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    assert a[0, 0] == a[0, 1]
    assert (a[1] == 0).all()


def plane_events(offset=0):
    x, y = np.meshgrid(np.arange(32), np.arange(32))
    x = x.ravel()
    y = y.ravel()
    t = offset + 1000 + x * 30 + y * 20
    order = np.argsort(t, kind="stable")
    return x[order], y[order], t[order], np.ones(len(t), dtype=int)


@pytest.mark.parametrize("split", [1, 17, 256])
def test_surface_chunk_and_large_timestamp_invariance(split):
    row = dict(
        window_start_us=0,
        window_end_us=4000,
        roi_x0=0.0,
        roi_y0=0.0,
        roi_x1=32.0,
        roi_y1=32.0,
        event_x_offset_px=0.0,
        polarity_encoding="zero_one",
    )
    event = plane_events()
    base = geometry_window([event], row)
    chunks = [tuple(v[i : i + split] for v in event) for i in range(0, len(event[0]), split)]
    changed = geometry_window(chunks, row)
    np.testing.assert_array_equal(base[0], changed[0])
    assert base[1] == changed[1]
    offset = 10**16
    shifted = geometry_window(
        [plane_events(offset)], {**row, "window_start_us": offset, "window_end_us": offset + 4000}
    )
    np.testing.assert_array_equal(base[0], shifted[0])


def test_exclusive_end_empty_and_polarity_validation():
    row = dict(
        window_start_us=0,
        window_end_us=4,
        roi_x0=0.0,
        roi_y0=0.0,
        roi_x1=4.0,
        roi_y1=4.0,
        event_x_offset_px=0.0,
        polarity_encoding="zero_one",
    )
    values, diagnostic = geometry_window([], row)
    assert values[11] == 0 and values[8] == 6 and values[9] == 10 and not diagnostic["valid"]
    with pytest.raises(ValueError):
        geometry_window([(np.array([1]), np.array([1]), np.array([4]), np.array([1]))], row)
    with pytest.raises(ValueError):
        geometry_window([(np.array([1]), np.array([1]), np.array([3]), np.array([-1]))], row)


def test_isotropic_reduced_support_and_aperture_failure():
    theta = np.arange(256) * 2 * np.pi / 256
    n = np.column_stack([np.cos(theta), np.sin(theta)])
    xy = 0.3 * n
    s = 0.6 * np.sum(n * xy, 1) + n @ np.array([0.2, -0.1])
    fit = fit_affine_normal_flow(xy, n, s)
    assert fit.valid and fit.mode == "isotropic"
    assert fit.kappa == pytest.approx(0.6)
    n[:] = [1, 0]
    assert not fit_affine_normal_flow(xy, n, s).valid


def test_cross_window_physical_transport():
    a = (np.zeros(12), dict(valid=True, raw=[0.5, 0, 0, 0, 0, 0], snapshot_us=100000))
    b = (np.zeros(12), dict(valid=True, raw=[0.5 / (1 - 0.05), 0, 0, 0, 0, 0], snapshot_us=200000))
    vector, reason = combine_windows(a, b)
    assert reason == "valid" and abs(vector[-1]) < 1e-12


def test_exact_sequence_probability_and_omission():
    d = exact_sequence_diagnostic(np.arange(-9.0, 0.0))
    assert d["sequence_count_vectors"] == 24310
    assert d["fraction_negative"] == pytest.approx(1)
    assert not d["retraining_performed"]


def test_missing_diagnostics_is_not_negative_branch():
    protocol = json.loads(
        Path("configs/protocol/scientific_recovery_v9_stage66_69.json").read_text()
    )
    with pytest.raises(DecisionBlocked):
        stage_decision("67", dict(execution_complete=False), protocol)
