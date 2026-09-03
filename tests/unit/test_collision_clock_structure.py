from __future__ import annotations

import inspect

import torch

from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.models.collision_clock_features import (
    FORBIDDEN_HEIGHT_MODULE_NAMES,
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_ttc import CollisionClockConfig, X0HeightBypassDirectPhase


def _model(mode: str) -> X0HeightBypassDirectPhase:
    config = CollisionClockConfig(
        encoder_hidden_dim=16,
        encoder_token_dim=8,
        residual_depth=1,
        clock_hidden_dim=8,
        dropout=0.0,
        motion_feature_mode=mode,  # type: ignore[arg-type]
    )
    encoder = HeightBypassEndpointEncoder(
        HeightBypassEncoderConfig(
            in_channels=12,
            hidden_dim=16,
            token_dim=8,
            residual_depth=1,
            dropout=0.0,
        )
    )
    return X0HeightBypassDirectPhase(encoder, config)


def test_base_dyn_topology_parameters_and_initialization_are_identical() -> None:
    torch.manual_seed(7)
    base = _model("global_uniform_zeroed_control")
    torch.manual_seed(7)
    dynamic = _model("global_uniform")
    assert module_topology_sha256(base) == module_topology_sha256(dynamic)
    assert tensor_state_sha256(base) == tensor_state_sha256(dynamic)
    assert [tuple(value.shape) for value in base.parameters()] == [
        tuple(value.shape) for value in dynamic.parameters()
    ]
    assert sum(value.numel() for value in base.parameters()) == sum(
        value.numel() for value in dynamic.parameters()
    )


def test_base_computes_matcher_then_zeroes_only_consumed_slots() -> None:
    torch.manual_seed(7)
    base = _model("global_uniform_zeroed_control").eval()
    torch.manual_seed(7)
    dynamic = _model("global_uniform").eval()
    inputs = torch.randn(2, 3, 12, 16, 16)
    delta = torch.full((2, 2), 0.05)
    with torch.no_grad():
        base_output = base(inputs, delta)
        dynamic_output = dynamic(inputs, delta)
    torch.testing.assert_close(
        base_output.diagnostics["global_transport_01_observed"],
        dynamic_output.diagnostics["global_transport_01_observed"],
    )
    assert torch.count_nonzero(base_output.diagnostics["global_transport_01_consumed"]) == 0
    assert torch.count_nonzero(base_output.diagnostics["global_transport_12_consumed"]) == 0
    torch.testing.assert_close(
        dynamic_output.diagnostics["global_transport_01_consumed"],
        dynamic_output.diagnostics["global_transport_01_observed"],
    )


def test_module_tree_and_forward_signature_exclude_forbidden_inputs() -> None:
    model = _model("global_uniform")
    module_names = tuple(name.lower() for name, _module in model.named_modules())
    for forbidden in FORBIDDEN_HEIGHT_MODULE_NAMES:
        assert not any(forbidden in name for name in module_names)
    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == ("inputs", "delta_t_s")
    forbidden_inputs = {"target", "bbox", "sequence", "track", "fold", "bucket"}
    assert forbidden_inputs.isdisjoint(signature.parameters)


def test_prefix_endpoint_encoding_is_independent_of_future_endpoint() -> None:
    model = _model("global_uniform").eval()
    prefix = torch.randn(1, 2, 12, 16, 16)
    first = torch.cat((prefix, torch.randn(1, 1, 12, 16, 16)), dim=1)
    second = torch.cat((prefix, torch.randn(1, 1, 12, 16, 16) * 100.0), dim=1)
    with torch.no_grad():
        _dense_a, token_a = model.encoder(first[:, :2].reshape(2, 12, 16, 16))
        _dense_b, token_b = model.encoder(second[:, :2].reshape(2, 12, 16, 16))
    torch.testing.assert_close(token_a, token_b, rtol=0.0, atol=0.0)


def test_fault_injected_historical_foreground_and_extent_are_never_called(monkeypatch) -> None:
    import e_jepa_ttc.models.causal_scale_ttc as historical

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden historical height/foreground path was called")

    monkeypatch.setattr(historical.CausalScaleTTC, "forward", forbidden)
    monkeypatch.setattr(historical, "soft_vertical_extent_from_logits", forbidden)
    model = _model("global_uniform").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 12, 16, 16), torch.full((1, 2), 0.05))
    assert output.benchmark_phase_mean.shape == (1,)
