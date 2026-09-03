from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml

from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.evaluation.collision_clock_runner import _direct_model
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


def test_production_base_dyn_identity_remains_exactly_audited() -> None:
    root = Path(__file__).resolve().parents[2]
    config_root = root / "configs/experiment/scientific_recovery_v9_eclock"
    configs = [
        yaml.safe_load((config_root / name).read_text(encoding="utf-8"))
        for name in ("x0_base_u.yaml", "x0_dyn_u.yaml")
    ]
    models = []
    for config in configs:
        torch.manual_seed(7)
        models.append(_direct_model(config))
    base, dynamic = models
    expected_topology = "0b0e08ceb586d05cebaaad7d6ec88ee9b2ee99de5978694680719daec2024504"
    expected_initialization = "363477531a257d5e7342234c5e78a1a56c89735dba487cbd3f70be209ee4bcad"
    expected_feature_schema = "d474b35a8830cfa1399ada2d6febf80e82ddb12194f1cc7159ca149aee18a164"
    assert type(base) is type(dynamic) is X0HeightBypassDirectPhase
    assert base.input_dim == dynamic.input_dim == 946
    assert sum(parameter.numel() for parameter in base.parameters()) == 308_005
    assert sum(parameter.numel() for parameter in dynamic.parameters()) == 308_005
    assert module_topology_sha256(base) == module_topology_sha256(dynamic) == expected_topology
    assert tensor_state_sha256(base) == tensor_state_sha256(dynamic) == expected_initialization
    assert base.feature_schema.manifest()["schema_sha256"] == expected_feature_schema
    assert dynamic.feature_schema.manifest()["schema_sha256"] == expected_feature_schema
    assert base.config.transport_radius == dynamic.config.transport_radius == 1
    assert base.config.transport_temperature == dynamic.config.transport_temperature == 0.02


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


def test_each_forward_makes_four_frozen_matcher_calls(monkeypatch) -> None:
    import e_jepa_ttc.models.collision_clock_motion as motion

    original = motion.local_correlation_match
    calls: list[tuple[int, float, bool]] = []

    def counted(*args, **kwargs):
        calls.append((kwargs["radius"], kwargs["temperature"], kwargs["return_probability"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(motion, "local_correlation_match", counted)
    model = _model("global_uniform").eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 12, 16, 16), torch.full((1, 2), 0.05))
    assert calls == [(1, 0.02, False)] * 4


def test_module_tree_and_forward_signature_exclude_forbidden_inputs() -> None:
    model = _model("global_uniform")
    module_names = tuple(name.lower() for name, _module in model.named_modules())
    for forbidden in FORBIDDEN_HEIGHT_MODULE_NAMES:
        assert not any(forbidden in name for name in module_names)
    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == ("inputs", "delta_t_s")
    forbidden_inputs = {
        "target_ttc",
        "benchmark_phase_target",
        "sample_weight",
        "sequence_id",
        "track_id",
        "outer_fold",
        "bucket",
        "bbox",
        "category",
        "checkpoint_family",
        "reference_family",
    }
    assert forbidden_inputs.isdisjoint(signature.parameters)
    assert tuple(signature.parameters) == ("inputs", "delta_t_s")


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
    monkeypatch.setattr(historical._EndpointEncoder, "forward", forbidden)
    monkeypatch.setattr(historical, "soft_vertical_extent_from_logits", forbidden)
    monkeypatch.setattr(historical, "log_ratio_to_inverse_ttc", forbidden)
    model = _model("global_uniform").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 12, 16, 16), torch.full((1, 2), 0.05))
    assert output.benchmark_phase_mean.shape == (1,)


def test_fault_injection_excludes_all_registered_geometry_heads() -> None:
    model = _model("global_uniform")
    registered = {name.lower() for name, _module in model.named_modules()}
    forbidden = {
        "foreground",
        "height_correction_head",
        "pair_projector",
        "uncertainty_head",
        "auxiliary_inverse_ttc_head",
        "transport_projector",
        "transport_router",
        "geometry_tokens",
        "visible_height",
        "visible_width",
        "analytic_log_height_ratio",
    }
    assert all(not any(item in name for name in registered) for item in forbidden)
