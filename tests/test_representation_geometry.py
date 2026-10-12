"""Geometry calibration contracts over real cached inputs; no optimizer updates."""

import copy

import numpy as np
import pytest
import torch

from e_jepa_ttc.range_readout.readout import VectorReadout
from e_jepa_ttc.rgb_port.features import pair133
from operational.representation_audit.geometry import initialize
from operational.representation_audit.precision import objective
from operational.representation_audit.run import OLD


def values():
    with np.load(OLD / "vector_cache/P/part_000000.npz", allow_pickle=False) as arc:
        return {k: torch.from_numpy(arc[k][:8].copy()) for k in arc.files if k != "sample_token"}


def test_constant_height_shift_preserves_ttc_and_native_scorer_gradients():
    calibrated, config, *_ = initialize(7, "Q0", device="cpu")
    native = VectorReadout(copy.deepcopy(calibrated.model))
    data = values()
    inputs = {k: v for k, v in data.items() if not k.startswith(("target_", "dino_"))}
    calibrated.eval()
    native.eval()
    before, after = native(inputs), calibrated(inputs)
    for name in (
        "ttc_mean_seconds",
        "pair_log_height_ratio",
        "collision_logits",
        "ttc_log_variance",
    ):
        assert torch.equal(getattr(before, name), getattr(after, name))
    a, _ = objective(native, data, config, "U0")
    b, _ = objective(calibrated, data, config, "U0")
    ga = torch.autograd.grad(a, [p for p in native.parameters() if p.requires_grad])
    gb = torch.autograd.grad(b, [p for p in calibrated.parameters() if p.requires_grad])
    assert all(torch.equal(x, y) for x, y in zip(ga, gb, strict=True))


def test_active_geometry_keeps_real_pair133_bit_identical():
    model, *_ = initialize(7, "Q1", device="cpu")
    model.eval()
    native = VectorReadout(copy.deepcopy(model.model)).eval()
    data = {k: v for k, v in values().items() if not k.startswith(("target_", "dino_"))}
    with torch.no_grad():
        before, after = native(data), model(data)
    assert not torch.equal(before.analytic_log_height_ratio, after.analytic_log_height_ratio)
    assert torch.equal(before.geometry_tokens, after.geometry_tokens)
    assert torch.equal(before.pair_tokens, after.pair_tokens)
    assert torch.equal(before.log_ratio_log_variance, after.log_ratio_log_variance)
    for output in (before, after):
        support = output.sensor_support
        supports = torch.stack((support[:, -1], support[:, -2:].min(-1).values), -1)
        actual = pair133(output.pair_tokens[:, -1], data["delta"][:, -1], supports)
        if output is before:
            reference = actual
        else:
            assert torch.equal(reference, actual)
    assert all(
        name.startswith("model.residual.scorer.")
        for name, p in model.named_parameters()
        if p.requires_grad
    )


def test_geometry_operator_serializes_and_rejects_targets():
    model, *_ = initialize(7, "Q1", device="cpu")
    restored, *_ = initialize(7, "Q0", device="cpu")
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    data = values()
    with pytest.raises(ValueError, match="targets"):
        model(data)
    inputs = {k: v for k, v in data.items() if not k.startswith(("target_", "dino_"))}
    assert torch.equal(model(inputs).ttc_mean_seconds, restored(inputs).ttc_mean_seconds)
    assert "height_coefficient" in model.state_dict()
    with pytest.raises(RuntimeError):
        VectorReadout(copy.deepcopy(model.model)).load_state_dict(model.state_dict(), strict=True)
