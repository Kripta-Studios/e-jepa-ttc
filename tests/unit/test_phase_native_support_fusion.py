"""Support features preserve IQ routing and fit masks only on clean training rows."""

import numpy as np
import pytest
import torch

from operational.phase_native_support.model import NativeSupportRouter
from operational.phase_native_support.training import (
    fit_preprocessing,
    qa_corruption,
    select_candidate,
)
from operational.phase_startup.model import ExpertRouter


@pytest.mark.parametrize(("mode", "parameters"), [("F2", 868), ("Fcap", 878)])
def test_capacities_and_original_router_parity_with_added_paths_disabled(mode, parameters):
    torch.manual_seed(7)
    original = ExpertRouter("quality", "inverse")
    original.gate[2].weight.data.normal_(0, 0.1)
    model = NativeSupportRouter(mode)
    assert sum(p.numel() for p in model.parameters()) == parameters
    with torch.no_grad():
        model.gate[0].weight.zero_()
        model.gate[0].bias.zero_()
        model.gate[2].weight.zero_()
        model.gate[0].weight[:18, :33].copy_(original.gate[0].weight)
        model.gate[0].bias[:18].copy_(original.gate[0].bias)
        model.gate[2].weight[:, :18].copy_(original.gate[2].weight)
        model.gate[2].bias.copy_(original.gate[2].bias)
    native = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [float("nan"), 2.0, 4.0], [0.0, 2.0, -3.0]]
    )
    geometry = torch.tensor([2.5, -2.5, 3.0, 2.0])
    quality = torch.zeros(4, 20)
    quality[:, 9] = 0.5
    support = torch.rand(4, 10)
    torch.testing.assert_close(
        model(native, geometry, quality, support),
        original(native, geometry, quality),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        model.base_inputs(native, geometry, quality),
        original.inputs(native, geometry, quality),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ["F2", "Fcap"])
def test_sign_equivariance_exact_fallback_and_script_roundtrip(mode):
    model = NativeSupportRouter(mode).eval()
    model.gate[2].weight.data.normal_(0, 0.1)
    native = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [float("nan"), 2.0, 3.0]])
    geometry = torch.tensor([2.5, 2.5, 2.5])
    quality = torch.ones(3, 20)
    quality[1:, 9] = 0
    support = torch.rand(3, 10)
    value = model(native, geometry, quality, support)
    assert value[1] == native[1, 0] and torch.isnan(value[2])
    negative_quality = quality.clone()
    negative_quality[:, :5] *= -1
    negative = model(-native, -geometry, negative_quality, support)
    torch.testing.assert_close(value, -negative, equal_nan=True)
    torch.testing.assert_close(
        torch.jit.script(model)(native, geometry, quality, support), value, equal_nan=True
    )


def test_constant_support_mask_uses_train_only_and_does_not_mask_original_inputs():
    model = NativeSupportRouter("F2")
    native, geometry, quality = torch.ones(4, 3), torch.ones(4), torch.ones(4, 20)
    support = torch.zeros(4, 10)
    support[:3, 1] = torch.tensor([0.0, 1.0, 2.0])
    support[3, 0] = 1000  # Held-out variation must not enable coordinate0.
    fit_preprocessing(model, native[:3], geometry[:3], quality[:3], support[:3])
    assert model.support_mask.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    first = model.normalized_inputs(native[3:], geometry[3:], quality[3:], support[3:])
    assert first[0, 33] == 0
    assert model.center.shape == (43,) and model.scale.shape == (43,)
    # Original33 normalization must retain exactly the historical robust contract.
    from operational.phase_robust.model import fit_normalization

    center, scale = fit_normalization(
        model.base_inputs(native[:3], geometry[:3], quality[:3]).numpy()
    )
    np.testing.assert_array_equal(model.center[:33].numpy(), center)
    np.testing.assert_array_equal(model.scale[:33].numpy(), scale)


def test_original_qa_corruption_draws_exactly_and_does_not_touch_support():
    native = torch.arange(1, 301, dtype=torch.float32).reshape(-1, 3)
    torch.manual_seed(7)
    expected = native.clone()
    indices = (torch.rand(len(native)) < 0.25).nonzero().flatten()
    experts = torch.randint(0, 3, (len(indices),))
    factors = torch.tensor([-3.0, -1.0, 0.25, 4.0])[torch.randint(0, 4, (len(indices),))]
    expected[indices, experts] *= factors
    torch.manual_seed(7)
    actual = qa_corruption(native)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.data_ptr() != native.data_ptr()


def test_selection_requires_all_gates_and_ties_prefer_f2():
    def report(value):
        return {
            d: {
                "coverage": 1.0,
                "macro_record_mae_seconds": value,
                "macro_record_relative_error_pct": value,
            }
            for d in ("all", "road", "slider")
        }

    reports = {"IQ": report(2.0), "F2": report(1.0), "Fcap": report(1.0)}
    result = select_candidate(reports)
    assert result["selected"] == "F2" and not result["support_dominates_capacity"]
    reports["F2"]["road"]["macro_record_mae_seconds"] = 0.9
    assert select_candidate(reports)["support_dominates_capacity"]
    reports["F2"]["slider"]["macro_record_mae_seconds"] = 3.0
    result = select_candidate(reports)
    assert result["selected"] == "Fcap" and not result["support_dominates_capacity"]
    reports["Fcap"]["road"]["coverage"] = 0.99
    assert select_candidate(reports)["selected"] is None


def test_support_table_contract_unknown_context_and_identity():
    from scripts.train_phase_native_support_fusion import validate_support_table

    table = {
        "token": np.array(["a", "b"]),
        "group": np.array(["g", "g"]),
        "quality10": np.array([[0.2, 0.0, 0.1, 0.2, 0.5] * 2, [0.0] * 10], np.float32),
        "knownmask2": np.array([[True, True], [False, False]]),
        "status": np.array(["fused", "unavailable"]),
        "reason": np.array(["ok", "absent"]),
    }
    validate_support_table(table, 2)
    for field, bad in (
        ("token", np.array(["a", "a"])),
        ("quality10", np.full((2, 10), np.nan, np.float32)),
        ("knownmask2", np.ones((2, 2), bool)),
    ):
        with pytest.raises(ValueError):
            validate_support_table({**table, field: bad}, 2)


def test_support_join_and_fold_reject_swapped_groups_and_leakage():
    from scripts.train_phase_native_support_fusion import attach_support, family_masks

    arrays = {
        "token": np.array(["a", "b", "c"]),
        "group": np.array(["g1", "g1", "g2"]),
        "family": np.array(["A", "A", "B"]),
    }
    table = {
        "token": arrays["token"].copy(),
        "group": arrays["group"].copy(),
        "quality10": np.zeros((3, 10), np.float32),
    }
    attach_support(arrays, table)
    train, test = family_masks(arrays, "B")
    assert train.tolist() == [True, True, False] and test.tolist() == [False, False, True]
    table["group"][0] = "g2"
    with pytest.raises(ValueError):
        attach_support(arrays, table)
    arrays["group"][0] = "g2"
    with pytest.raises(ValueError):
        family_masks(arrays, "B")


@pytest.mark.parametrize("mode", ["F2", "Fcap"])
def test_exact_inverse_cancellation_keeps_native_f0(mode):
    model = NativeSupportRouter(mode)
    model.gate[2].bias.data.zero_()
    native = torch.tensor([[1.0, -1.0, float("nan")]])
    geometry = torch.tensor([float("nan")])
    quality = torch.ones(1, 20)
    assert model(native, geometry, quality, torch.zeros(1, 10)).item() == 1.0


def test_failed_support_seal_cannot_open_targets(monkeypatch, tmp_path):
    from argparse import Namespace

    from scripts import train_phase_native_support_fusion as runner

    def reject(*_args):
        raise ValueError("support cache hash changed")

    def targets(*_args):
        pytest.fail("Target join reached before cache admission")

    monkeypatch.setattr(runner, "verify_support_cache", reject)
    monkeypatch.setattr(runner, "load_training_data", targets)
    with pytest.raises(ValueError, match="hash changed"):
        runner.load_experiment_data(Namespace(support_cache=tmp_path, population=tmp_path))


def test_wrong_support_identity_cannot_open_targets(monkeypatch, tmp_path):
    from argparse import Namespace

    from scripts import train_phase_native_support_fusion as runner

    path = tmp_path / "population.npz"
    np.savez(path, token=np.array(["real"]), group=np.array(["group"]))
    monkeypatch.setattr(
        runner,
        "verify_support_cache",
        lambda *_: {
            "DEVELOPMENT": {
                "token": np.array(["wrong"]),
                "group": np.array(["group"]),
                "quality10": np.zeros((1, 10), np.float32),
            }
        },
    )

    def targets(*_args):
        pytest.fail("Target join reached before identity admission")

    monkeypatch.setattr(runner, "load_training_data", targets)
    with pytest.raises(ValueError, match="identity differs"):
        runner.load_experiment_data(Namespace(support_cache=tmp_path, population=path))
