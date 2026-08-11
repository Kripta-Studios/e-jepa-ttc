from __future__ import annotations

import inspect

import torch

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig


def _model(mode: str) -> CausalScaleTTC:
    torch.manual_seed(123)
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
            foreground_temporal_smoothing=0.15,
            foreground_temporal_smoothing_mode=mode,  # type: ignore[arg-type]
        )
    ).eval()


def test_neural_forward_accepts_only_event_tensor_delta_and_diagnostic_flag() -> None:
    signature = inspect.signature(CausalScaleTTC.forward)
    assert tuple(signature.parameters) == (
        "self",
        "inputs",
        "delta_t_s",
        "return_dense_features",
    )


def test_legacy_smoothing_is_deliberately_not_prefix_invariant() -> None:
    model = _model("symmetric_legacy")
    x = torch.randn(1, 3, 2, 32, 32)
    dt2 = torch.full((1, 1), 0.1)
    dt3 = torch.full((1, 2), 0.1)
    with torch.inference_mode():
        prefix = model(x[:, :2], dt2)
        full = model(x, dt3)
    # t0 also receives its next endpoint in the legacy symmetric filter.
    assert not torch.allclose(prefix.foreground_logits[:, :2], full.foreground_logits[:, :2])


def test_causal_left_smoothing_is_prefix_invariant_for_all_emitted_pair_state() -> None:
    model = _model("causal_left")
    x = torch.randn(1, 4, 2, 32, 32)
    with torch.inference_mode():
        prefix = model(x[:, :3], torch.full((1, 2), 0.1))
        full = model(x, torch.full((1, 3), 0.1))

    torch.testing.assert_close(prefix.foreground_logits, full.foreground_logits[:, :3])
    torch.testing.assert_close(prefix.geometry_tokens, full.geometry_tokens[:, :3])
    torch.testing.assert_close(prefix.pair_tokens, full.pair_tokens[:, :2])
    torch.testing.assert_close(prefix.analytic_log_height_ratio, full.analytic_log_height_ratio[:, :2])
    torch.testing.assert_close(prefix.residual_log_height_ratio, full.residual_log_height_ratio[:, :2])
    torch.testing.assert_close(prefix.pair_log_height_ratio, full.pair_log_height_ratio[:, :2])
    torch.testing.assert_close(prefix.pair_ttc_seconds, full.pair_ttc_seconds[:, :2])


def test_common_roi_ignores_boxes_outside_explicit_t1_t2_indices() -> None:
    boxes = [
        [10.0, 10.0, 20.0, 20.0],
        [20.0, 20.0, 40.0, 50.0],
        [21.0, 18.0, 48.0, 55.0],
        [1.0, 1.0, 999.0, 999.0],
    ]
    baseline = common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25)
    mutated = [row[:] for row in boxes]
    mutated[0] = [-500.0, -500.0, -400.0, -400.0]
    mutated[3] = [5000.0, 5000.0, 9000.0, 9000.0]
    assert common_square_from_boxes(mutated, (1, 2), margin_fraction=0.25) == baseline


def test_common_roi_is_oracle_endpoint_box_conditioned() -> None:
    boxes = [
        [10.0, 10.0, 20.0, 20.0],
        [20.0, 20.0, 40.0, 50.0],
        [21.0, 18.0, 48.0, 55.0],
    ]
    baseline = common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25)
    boxes[2] = [80.0, 80.0, 110.0, 130.0]
    assert common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25) != baseline


def test_a7_transport_encoder_copy_is_separate_and_trainable_when_primary_frozen() -> None:
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
            transport_enabled=True,
            transport_radius=1,
            transport_temperature=0.02,
            transport_encoder_copy_enabled=True,
        )
    )
    assert model.transport_encoder is not None
    model.transport_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.transport_encoder.parameters())

    x = torch.randn(2, 3, 2, 32, 32)
    out = model(x, torch.full((2, 2), 0.1))
    out.pair_log_height_ratio.square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.transport_encoder.parameters())
