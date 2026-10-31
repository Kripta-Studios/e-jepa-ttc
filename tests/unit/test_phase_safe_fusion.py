"""Physical and deployment contracts for sign-safe research fusion."""

import numpy as np
import pytest
import torch

from operational.phase_safe_fusion.model import SafeFusion


def test_progress_snapshots_do_not_overwrite_completed_fold(tmp_path):
    from scripts.train_phase_safe_fusion import save_progress

    save_progress(tmp_path, "family1", "LG", {"LG": np.array([1.0, np.nan])})
    save_progress(tmp_path, "family2", "LG", {"LG": np.array([1.0, 2.0])})
    with np.load(tmp_path / "progress/family1/LG.npz") as first:
        assert np.isnan(first["LG"][1])


def test_supervision_can_train_a_wrong_singleton_sign():
    from operational.phase_safe_fusion.training import gate_risk

    model = SafeFusion("log")
    native = torch.tensor([[-2.0, 3.0, 4.0]])
    geo = torch.tensor([5.0])
    quality = torch.zeros(1, 20)
    quality[:, 9] = 0.2
    target = torch.tensor([3.0])
    model(native, geo, quality).sum().backward()
    assert sum(p.grad.abs().sum() for p in model.features.gate.parameters()) == 0
    model.zero_grad()
    loss = gate_risk(model, native, geo, quality, target, torch.tensor(3.0)).sum()
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.features.gate.parameters()) > 0


def test_seed_aggregation_avoids_cancellation_and_preserves_fallback():
    from operational.phase_safe_fusion.runtime import combine_seeds

    native = torch.tensor([[47.665657, 1.0, 1.0], [-6.0, 3.0, 3.0]])
    quality = torch.zeros(2, 20)
    quality[1, 9] = 0.2
    predictions = torch.tensor([[47.665657, -6.0], [47.665657, 3.0], [47.665657, 3.0]])
    result = combine_seeds(predictions, native, quality)
    assert torch.equal(result[:1], native[:1, 0])
    assert result[1] == 3.0
    assert torch.equal(combine_seeds(-predictions, -native, quality), -result)


@pytest.mark.parametrize("mode", ["log", "residual"])
def test_trained_feature_sign_equivariance(mode):
    torch.manual_seed(17)
    model = SafeFusion(mode)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
        model.features.center.normal_()
        model.features.scale.uniform_(0.2, 2.0)
    native, geo, quality = inputs()
    quality[:, :5] = torch.randn(2, 5)
    reversed_quality = quality.clone()
    reversed_quality[:, :5] *= -1
    assert torch.equal(model(-native, -geo, reversed_quality), -model(native, geo, quality))


def test_selection_rejects_any_coverage_or_domain_regression():
    from scripts.train_phase_safe_fusion import select_candidate

    def result(rae, mae, coverage=1.0):
        return {
            name: {
                "macro_record_relative_error_pct": rae,
                "macro_record_mae_seconds": mae,
                "coverage": coverage,
            }
            for name in ("road", "slider", "all")
        }

    reports = {
        "IQ": result(20, 1),
        "LG": result(10, 0.9),
        "LA": result(9, 0.8, 0.99),
        "GR": result(8, 1.1),
    }
    assert select_candidate(reports) == "LG"
    reports["LG"]["slider"]["macro_record_mae_seconds"] = 1.01
    assert select_candidate(reports) is None


def inputs():
    native = torch.tensor([[-42.225, -26.359, 7.286], [2.0, 2.0, 2.0]])
    geometry = torch.tensor([2.946, 1.0])
    quality = torch.zeros(2, 20)
    quality[:, 9] = 0.25
    return native, geometry, quality


def test_opposite_signs_cannot_create_cancellation_outlier():
    model = SafeFusion("log")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    native, geo, quality = inputs()
    result = model(native, geo, quality)
    # Equal sign masses choose first available expert sign, with bounded magnitude.
    assert -42.226 <= result[0] <= -26.358
    assert 1 <= result[1] <= 2


@pytest.mark.parametrize("mode", ["log", "residual"])
def test_sign_reversal_and_script_export(mode):
    model = SafeFusion(mode).eval()
    native, geo, quality = inputs()
    assert torch.equal(model(-native, -geo, quality), -model(native, geo, quality))
    scripted = torch.jit.script(model)
    assert torch.equal(model(native, geo, quality), scripted(native, geo, quality))


def test_fallback_exact_for_absent_stale_or_invalid_history():
    native, geo, quality = inputs()
    for bad in (0.0, float("nan")):
        quality[:, 9] = bad
        assert torch.equal(SafeFusion("log")(native, geo, quality), native[:, 0])
    quality[:, 9] = 0.25
    quality[:, 11] = 0.151
    assert torch.equal(SafeFusion("residual")(native, geo, quality), native[:, 0])


def test_geometry_residual_is_bounded_and_missing_geometry_uses_native():
    model = SafeFusion("residual")
    native, geo, quality = inputs()
    with torch.no_grad():
        model.correction.bias.fill_(100)
    result = model(native, geo, quality)
    assert torch.all(result <= geo * 2.00001)
    assert torch.all(result >= geo * 0.49999)
    missing = model(native, torch.full_like(geo, float("nan")), quality)
    assert torch.isfinite(missing).all()


@pytest.mark.parametrize("mode", ["log", "residual"])
def test_invalid_experts_do_not_poison_outputs_or_gradients(mode):
    model = SafeFusion(mode)
    native, geo, quality = inputs()
    native[0, 1] = float("nan")
    geo[0] = float("nan")
    result = model(native, geo, quality)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    native[:] = float("nan")
    geo[:] = float("nan")
    assert torch.isnan(model(native, geo, quality)).all()
