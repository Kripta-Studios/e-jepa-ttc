from __future__ import annotations

import inspect

import torch

from e_jepa_ttc.models.collision_clock_features import (
    ClockFeatureSchema,
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
    assemble_x0_clock_features,
)
from e_jepa_ttc.models.collision_clock_motion import GLOBAL_TRANSPORT_FEATURE_NAMES
from e_jepa_ttc.models.collision_clock_ttc import X0HeightBypassDirectPhase


def test_forward_boundary_contains_only_events_and_timing() -> None:
    parameters = tuple(inspect.signature(X0HeightBypassDirectPhase.forward).parameters)
    assert parameters == ("self", "inputs", "delta_t_s")
    source = inspect.getsource(X0HeightBypassDirectPhase.forward).lower()
    for forbidden in ("target", "bbox", "sequence", "track", "fold", "bucket", "sample_weight"):
        assert forbidden not in source


def test_feature_schema_is_exact_946_and_order_sensitive() -> None:
    schema = ClockFeatureSchema(128)
    assert schema.input_dim == 946
    assert schema.motion_names == GLOBAL_TRANSPORT_FEATURE_NAMES
    try:
        ClockFeatureSchema(128, motion_names=tuple(reversed(GLOBAL_TRANSPORT_FEATURE_NAMES)))
    except ValueError:
        pass
    else:
        raise AssertionError("reordered motion schema was accepted")


def test_temporal_feature_assembly_preserves_dynamic_values() -> None:
    schema = ClockFeatureSchema(4)
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    motion01 = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    motion12 = motion01 + 10.0
    vector = assemble_x0_clock_features(
        tokens,
        motion01,
        motion12,
        torch.full((2, 2), 0.05),
        torch.ones(2, 3),
        schema=schema,
    )
    start = 7 * schema.token_dim
    torch.testing.assert_close(vector[:, start : start + 9], motion01)
    torch.testing.assert_close(vector[:, start + 9 : start + 18], motion12)


def test_height_bypass_copy_registers_only_sanctioned_a5_modules() -> None:
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig

    source = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=8,
            geometry_dim=4,
            residual_depth=1,
            dropout=0.0,
        )
    )
    copied = HeightBypassEndpointEncoder.from_causal_scale_topology(
        source,
        config=HeightBypassEncoderConfig(
            in_channels=12,
            hidden_dim=8,
            token_dim=4,
            residual_depth=1,
            dropout=0.0,
        ),
    )
    assert set(copied.state_dict()) == {
        *(f"features.{name}" for name in source.encoder.features.state_dict()),
        *(f"token.{name}" for name in source.encoder.token.state_dict()),
    }
    assert all("foreground" not in name for name, _module in copied.named_modules())
