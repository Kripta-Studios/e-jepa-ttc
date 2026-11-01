import numpy as np
import torch

from operational.phase_robust.model import QualityFusion, fit_normalization
from operational.phase_robust.runtime import FusionEnsemble


def test_gate_preserves_missing_geometry_and_bounds_valid_mixture():
    model = QualityFusion("quality")
    native = torch.tensor([[2.0, 2.0, 3.0], [-2.0, -3.0, -2.0], [4.0, 4.0, 4.0]])
    geometry = torch.tensor([1.0, -1.0, float("nan")])
    quality = torch.zeros(3, 20)
    result = model(native, geometry, quality)
    assert 1 <= result[0] <= 2
    assert -2 <= result[1] <= -1
    assert result[2] == native[2, 0]
    opposite = model(native[:1], torch.tensor([-1.0]), quality[:1])
    assert opposite[0] == native[0, 0]


def test_missing_native_uses_supported_geometry_and_double_absence_stays_missing():
    model = QualityFusion("quality")
    native = torch.full((2, 3), float("nan"))
    result = model(native, torch.tensor([2.0, float("nan")]), torch.zeros(2, 20))
    assert result[0] == 2
    assert torch.isnan(result[1])


def test_controls_match_parameter_count_and_normalization_uses_only_fit_rows():
    a, b = QualityFusion("quality"), QualityFusion("native")
    assert sum(p.numel() for p in a.parameters()) == 401
    assert sum(p.numel() for p in b.parameters()) == 401
    x = np.array([[1.0, 2.0], [3.0, 4.0], [1e6, -1e6]])
    center, scale = fit_normalization(x[:2])
    np.testing.assert_allclose(center, [2, 3])
    assert np.isfinite(scale).all() and (scale > 0).all()


def test_export_matches_fallback_and_signed_outputs():
    model = QualityFusion("quality").eval()
    scripted = torch.jit.script(model)
    native = torch.tensor([[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0], [1.0, 1.0, 1.0]])
    geo = torch.tensor([1.0, -1.0, float("nan")])
    quality = torch.zeros(3, 20)
    torch.testing.assert_close(model(native, geo, quality), scripted(native, geo, quality))


def test_ensemble_fallback_is_bitwise_identical_for_arbitrary_native_values():
    torch.manual_seed(13)
    native = torch.rand(10000, 3) * 60
    ensemble = FusionEnsemble([QualityFusion() for _ in range(3)])
    out = ensemble(native, torch.full((10000,), float("nan")), torch.zeros(10000, 20))
    assert torch.equal(out, native[:, 0])
