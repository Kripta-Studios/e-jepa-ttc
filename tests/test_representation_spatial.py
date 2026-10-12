"""Global/spatial contrast preserves original producer tokens and PAIR inputs."""

import copy

import numpy as np
import pytest
import torch

from e_jepa_ttc.rgb_port.features import pair133
from operational.representation_audit.geometry import initialize
from operational.representation_audit.run import OLD
from operational.representation_audit.spatial_model import (
    SpatialReadout,
    fit_spatial_ridge,
    spatial_inputs,
)


def test_global_control_removes_vertical_order_but_active_keeps_it():
    raw = np.arange(2 * 3 * 512, dtype=np.float64).reshape(2, 3, 512)
    swapped = raw.reshape(2, 3, 64, 8)[:, :, :, ::-1].reshape(2, 3, 512)
    np.testing.assert_array_equal(spatial_inputs(raw, False), spatial_inputs(swapped, False))
    assert not np.array_equal(spatial_inputs(raw, True), spatial_inputs(swapped, True))
    assert not spatial_inputs(raw, False)[:, :, 64:].any()


@pytest.mark.parametrize("active", [False, True])
def test_chunked_ridge_matches_dense_registered_objective(active):
    rng = np.random.default_rng(941)
    raw = rng.normal(size=(193, 3, 512)).astype(np.float32)
    original = rng.uniform(0.3, 0.7, size=(193, 3))
    height = original * np.exp(rng.normal(0, 0.1, size=(193, 3)))
    valid = np.ones((193, 3), bool)
    valid[:, 0] = False
    analytic = rng.normal(0, 0.01, size=(193, 2))
    values = {
        "height_features": raw,
        "target_valid": valid,
        "target_height_normalized": height,
        "visible_height": original,
        "analytic": analytic,
    }
    fitted = fit_spatial_ridge(values, active, chunk_size=71)
    dense = spatial_inputs(raw.astype(np.float64), active)
    mean = dense[valid].mean(0)
    scale = np.maximum(dense[valid].std(0), 1e-6)
    z = (dense - mean) / scale
    x = z[valid]
    d = z[:, -1] - z[:, -2]
    y = np.log(height[valid]) - np.log(original[valid])
    target = np.log(height[:, -1] / height[:, -2]) - analytic[:, -1]
    factor = len(y) / len(target)
    matrix = x.T @ x / y.var() + factor * (d.T @ d) / target.var() + np.eye(512)
    rhs = x.T @ (y - y.mean()) / y.var() + factor * (d.T @ target) / target.var()
    beta = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(fitted["mean"], mean, atol=1e-14)
    np.testing.assert_allclose(fitted["scale"], scale, atol=1e-14)
    np.testing.assert_allclose(fitted["coefficient"], beta, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(
        fitted["unclipped_training_height_rmse"],
        np.sqrt(np.mean((x @ beta + y.mean() - y) ** 2)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fitted["unclipped_training_temporal_rmse"],
        np.sqrt(np.mean((d @ beta - target) ** 2)),
        atol=1e-10,
    )


def test_spatial_height_does_not_change_real_pair133_or_uncertainty():
    original, *_ = initialize(7, "Q1", device="cpu")
    calibration = {
        "role": "P",
        "mean": [0.0] * 512,
        "scale": [1.0] * 512,
        "coefficient": np.linspace(-0.01, 0.01, 512).tolist(),
        "intercept": 0.0,
    }
    control = SpatialReadout(copy.deepcopy(original.model), calibration, "SP0").eval()
    active = SpatialReadout(copy.deepcopy(original.model), calibration, "SP1").eval()
    with np.load(OLD / "vector_cache/P/part_000000.npz", allow_pickle=False) as arc:
        inputs = {
            k: torch.from_numpy(arc[k][:4].copy())
            for k in arc.files
            if k != "sample_token" and not k.startswith(("target_", "dino_"))
        }
    inputs["height_features"] = torch.from_numpy(
        np.random.default_rng(17).normal(size=(4, 3, 512)).astype(np.float32)
    )
    with torch.no_grad():
        a, b = control(inputs), active(inputs)
    assert not torch.equal(a.analytic_log_height_ratio, b.analytic_log_height_ratio)
    for name in ("pair_tokens", "geometry_tokens", "log_ratio_log_variance", "sensor_support"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    supports = torch.stack((a.sensor_support[:, -1], a.sensor_support[:, -2:].min(-1).values), -1)
    assert torch.equal(
        pair133(a.pair_tokens[:, -1], inputs["delta"][:, -1], supports),
        pair133(b.pair_tokens[:, -1], inputs["delta"][:, -1], supports),
    )
    assert all(
        name.startswith("model.residual.scorer.")
        for name, p in active.named_parameters()
        if p.requires_grad
    )
    with pytest.raises(ValueError, match="targets"):
        active({**inputs, "target_ttc": torch.ones(4)})
    control.load_state_dict(active.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(control(inputs).ttc_mean_seconds, active(inputs).ttc_mean_seconds)
