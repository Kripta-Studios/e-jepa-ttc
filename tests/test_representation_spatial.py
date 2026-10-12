"""Global/spatial contrast preserves original producer tokens and PAIR inputs."""

import copy

import numpy as np
import pytest
import torch

from e_jepa_ttc.rgb_port.features import pair133
from operational.representation_audit.geometry import initialize
from operational.representation_audit.run import OLD
from operational.representation_audit.spatial_model import SpatialReadout, spatial_inputs


def test_global_control_removes_vertical_order_but_active_keeps_it():
    raw = np.arange(2 * 3 * 512, dtype=np.float64).reshape(2, 3, 512)
    swapped = raw.reshape(2, 3, 64, 8)[:, :, :, ::-1].reshape(2, 3, 512)
    np.testing.assert_array_equal(spatial_inputs(raw, False), spatial_inputs(swapped, False))
    assert not np.array_equal(spatial_inputs(raw, True), spatial_inputs(swapped, True))
    assert not spatial_inputs(raw, False)[:, :, 64:].any()


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
