"""CPU admission of the static complete-transport adapter, without CUDA claims."""

from dataclasses import fields

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.local_transport import local_correlation_match, transport_physical_features
from operational.train40_runtime.compiled_transport import (
    TransportPairTensorCore,
    _rewritten_forward,
    attach_transport_pair_core,
)


def _reference(a, b, wa, wb, radius, deterministic):
    forward = local_correlation_match(a, b, radius=radius, temperature=0.02)
    reverse = local_correlation_match(b, a, radius=radius, temperature=0.02)
    return (
        transport_physical_features(
            forward,
            reverse,
            foreground_weight=wa,
            radius=radius,
            deterministic_sampling=deterministic,
        ),
        transport_physical_features(
            reverse,
            forward,
            foreground_weight=wb,
            radius=radius,
            deterministic_sampling=deterministic,
        ),
    )


@pytest.mark.parametrize("radius,size", [(1, 8), (2, 8), (4, 12)])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_complete_transport_core_preserves_original_summaries(radius, size, deterministic, reuse):
    torch.manual_seed(18)
    a, b = torch.randn(2, 8, size, size), torch.randn(2, 8, size, size)
    wa, wb = torch.rand(2, 1, size, size), torch.rand(2, 1, size, size)
    core = TransportPairTensorCore(
        radius, 0.02, deterministic_sampling=deterministic, reuse_reverse=reuse
    )
    for aa, bb in ((a, b), (a * 0, b * 0)):
        actual = core(aa, bb, wa, wb)
        expected = _reference(aa, bb, wa, wb, radius, deterministic)
        for x, y in zip(actual, expected, strict=True):
            if reuse:
                # Reversing a volume can change the kernel layout of channel
                # reductions. Grid12/r4 is a measured non-bit-exact case.
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
            else:
                assert torch.equal(x, y)


def _model(pyramid):
    return (
        CausalScaleTTC(
            CausalScaleTTCConfig(
                hidden_dim=8,
                geometry_dim=8,
                residual_depth=1,
                dropout=0.0,
                foreground_temporal_smoothing=0.15,
                foreground_temporal_smoothing_mode="causal_left",
                temporal_inverse_ttc_blend=0.75,
                transport_enabled=True,
                transport_radius=1,
                transport_mode="adaptive_pyramid" if pyramid else "legacy",
                transport_coarse_radius=1,
                deterministic_transport_sampling=True,
            )
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("pyramid", [False, True])
def test_complete_transport_model_view_preserves_all_outputs_guards_and_state(pyramid):
    torch.manual_seed(19)
    model = _model(pyramid)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_modules = dict(model.named_modules())
    view = attach_transport_pair_core(model)
    sensor, delta = torch.rand(3, 3, 12, 32, 32), torch.full((3, 2), 0.1)
    with torch.inference_mode():
        actual = view(sensor, delta, return_dense_features=True)
        expected = model(sensor, delta, return_dense_features=True)
        with pytest.raises(ValueError, match="strictly positive"):
            view(sensor, delta * 0)
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        elif isinstance(a, dict):
            assert all(torch.equal(a[key], b[key]) for key in a), field.name
        else:
            assert a == b
    assert dict(model.named_modules()) == original_modules
    assert all(
        torch.equal(value, view.state_dict()[name]) for name, value in original_state.items()
    )
    assert all(a is b for a, b in zip(model.parameters(), view.parameters(), strict=True))
    assert view._train40_transport_binding.replacements == 2


def test_complete_transport_has_one_dynamo_graph_without_inductor_or_cuda():
    torch.manual_seed(20)
    core = TransportPairTensorCore(1, 0.02, deterministic_sampling=True)
    compiled = torch.compile(core, backend="eager", fullgraph=True, dynamic=False)
    inputs = (torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8))
    weights = (torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    with torch.inference_mode():
        expected = core(*inputs, *weights)
        actual = compiled(*inputs, *weights)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_complete_transport_rejects_training_and_stacked_adapters():
    model = _model(False)
    with pytest.raises(ValueError, match="frozen eval"):
        attach_transport_pair_core(model.train())
    model.eval()
    with pytest.raises(ValueError, match="explicit backend"):
        attach_transport_pair_core(model, compile_options={})
    with pytest.raises(ValueError, match="already attached"):
        attach_transport_pair_core(attach_transport_pair_core(model))


def test_complete_transport_adapter_rejects_changed_frozen_statements(monkeypatch):
    import inspect

    original = inspect.getsource(CausalScaleTTC._forward_impl)
    changed = original.replace("radius=fine_radius,", "radius=fine_radius + 1,", 1)
    monkeypatch.setattr(inspect, "getsource", lambda _: changed)
    with pytest.raises(RuntimeError, match="two exact replacements"):
        _rewritten_forward()
