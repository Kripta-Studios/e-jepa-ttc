"""Nonlinear height stage isolation, zero-init control and native-map cache parity."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from operational.range_transfer.qa import same
from operational.representation_audit.export_weights import load_export
from operational.representation_audit.nonlinear import initialize
from operational.representation_audit.spatial_geometry import initialize as init_spatial
from tests.fixtures.representation_audit import install


@pytest.fixture
def audit_values(tmp_path, monkeypatch):
    return install(tmp_path, monkeypatch)


def inputs(values, count=4, targets=False):
    return {
        key: value[:count].clone()
        for key, value in values.items()
        if targets or not key.startswith(("target_", "dino_"))
    }


def test_zero_initialization_matches_sp1_and_scorer_state_exactly(audit_values):
    reference, _, opt_a, sched_a, rng_a = init_spatial(7, "SP1", device="cpu")
    candidate, _, opt_b, sched_b, rng_b = initialize(7, "NL1", device="cpu", technical=True)
    assert same(opt_a.state_dict(), opt_b.state_dict())
    assert same(sched_a.state_dict(), sched_b.state_dict())
    assert torch.equal(rng_a.get_state(), rng_b.get_state())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    reference.eval()
    candidate.eval()
    values = inputs(audit_values)
    left, right = reference(values), candidate(values)
    for key in (
        "ttc_mean_seconds",
        "pair_log_height_ratio",
        "pair_tokens",
        "log_ratio_log_variance",
    ):
        assert torch.equal(getattr(left, key), getattr(right, key))
    grads_a = torch.autograd.grad(
        left.pair_log_height_ratio.sum(), reference.model.residual.scorer.parameters()
    )
    grads_b = torch.autograd.grad(
        right.pair_log_height_ratio.sum(), candidate.model.residual.scorer.parameters()
    )
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b, strict=True))
    with pytest.raises(ValueError, match="targets"):
        candidate({**values, "target_ttc": torch.ones(4)})
    assert all(
        key.startswith("model.residual.scorer.")
        for key, p in candidate.named_parameters()
        if p.requires_grad
    )


def test_geometry_stage_uses_only_height_targets_and_only_trains_new_mlp(audit_values):
    model, _, optimizer, _, _ = initialize(7, "NLH", device="cpu", technical=True)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert all(
        key.startswith("height_mlp.") for key, p in model.named_parameters() if p.requires_grad
    )
    values = inputs(audit_values, targets=True)
    actual = model.geometry_loss(values)
    changed = model.geometry_loss({**values, "target_ttc": values["target_ttc"] * 1000})
    assert torch.equal(actual, changed)
    parameters = list(model.height_mlp.parameters())
    gradients = torch.autograd.grad(actual, parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0


def test_native_map_composition_and_cached_height_gradients_match(tmp_path, audit_values):
    model, *_ = initialize(7, "NLH", device="cpu", technical=True)
    values = inputs(audit_values, count=2, targets=True)
    with torch.no_grad():
        # These features were produced from the same synthetic events through the
        # native encoder and registered pooling operator in the shared fixture.
        raw = audit_values["height_features"][:2].clone()
        final = model.height_mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        final.weight.fill_(0.001)
    path = tmp_path / "prepared.npz"
    np.savez_compressed(path, height_features=raw.numpy())
    with np.load(path, allow_pickle=False) as archive:
        cached = torch.from_numpy(archive["height_features"].copy())
    direct_values = {**values, "height_features": raw}
    cached_values = {**values, "height_features": cached}
    assert torch.equal(model.height_delta(direct_values), model.height_delta(cached_values))
    direct_loss, cached_loss = (
        model.geometry_loss(direct_values),
        model.geometry_loss(cached_values),
    )
    assert torch.equal(direct_loss, cached_loss)
    params = list(model.height_mlp.parameters())
    direct_grad = torch.autograd.grad(direct_loss, params)
    cached_grad = torch.autograd.grad(cached_loss, params)
    assert all(torch.equal(a, b) for a, b in zip(direct_grad, cached_grad, strict=True))
    direct_inference = {
        k: v for k, v in direct_values.items() if not k.startswith(("target_", "dino_"))
    }
    cached_inference = {
        k: v for k, v in cached_values.items() if not k.startswith(("target_", "dino_"))
    }
    direct_output, cached_output = model(direct_inference), model(cached_inference)
    assert torch.equal(direct_output.ttc_mean_seconds, cached_output.ttc_mean_seconds)
    direct_ratio_grad = torch.autograd.grad(direct_output.pair_log_height_ratio.sum(), params)
    cached_ratio_grad = torch.autograd.grad(cached_output.pair_log_height_ratio.sum(), params)
    assert all(torch.equal(a, b) for a, b in zip(direct_ratio_grad, cached_ratio_grad, strict=True))
    baseline, *_ = init_spatial(7, "SP1", device="cpu")
    baseline.eval()
    with torch.no_grad():
        original = baseline(direct_inference)
    for key in ("pair_tokens", "geometry_tokens", "sensor_support", "log_ratio_log_variance"):
        assert torch.equal(getattr(original, key), getattr(direct_output, key))


def test_portable_nonlinear_operator_preserves_learned_height_branch(tmp_path, audit_values):
    model, *_ = initialize(7, "NLH", device="cpu", technical=True)
    model.eval()
    with torch.no_grad():
        final = model.height_mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.fill_(0.012)
    metadata = {
        "kind": "vector_nonlinear_height",
        "model_config": asdict(model.model.config),
        "operator": asdict(ResidualSpec(0.05)),
        "geometry_objective_variances": {
            "variance_height_target": model.height_variance,
            "variance_temporal_target": model.temporal_variance,
        },
    }
    path = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "v17_explicit_operator_v1",
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    restored = load_export(path)
    with torch.no_grad():
        left, right = model(inputs(audit_values)), restored(inputs(audit_values))
    assert torch.equal(left.ttc_mean_seconds, right.ttc_mean_seconds)
    assert torch.equal(left.pair_log_height_ratio, right.pair_log_height_ratio)
