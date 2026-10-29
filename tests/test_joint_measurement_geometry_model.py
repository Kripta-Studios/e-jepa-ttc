"""Geometry readout clock, affine decomposition, fallback and encoder-gradient contracts."""

import math

import torch

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.joint_measurement.geometry_model import (
    GeometricPairMeasurement,
    relative_scale_phase,
    weighted_affine,
)


def coordinates():
    axis = (torch.arange(32, dtype=torch.float64) + 0.5) / 16 - 1
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy))[None]


def test_joint_affine_fit_recovers_rotation_anisotropic_scale_and_translation():
    angle = 0.4
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    matrix = rotation @ torch.diag(torch.tensor([1.2, 0.8], dtype=torch.float64))
    shift = torch.tensor([0.07, -0.09], dtype=torch.float64)
    position = coordinates()
    flow = (
        torch.einsum("ij,bjhw->bihw", matrix - torch.eye(2), position) + shift[None, :, None, None]
    )
    weights = torch.ones_like(flow[:, 0])
    actual, translation = weighted_affine(flow, weights)
    torch.testing.assert_close(actual[0], matrix, atol=2e-4, rtol=0)
    torch.testing.assert_close(translation[0], shift, atol=1e-10, rtol=0)
    torch.testing.assert_close(
        actual.square().sum(1).sqrt()[0],
        torch.tensor([1.2, 0.8], dtype=torch.float64),
        atol=2e-4,
        rtol=0,
    )


def test_pure_translation_is_not_expansion_even_with_nonuniform_weights():
    position = coordinates()
    flow = torch.ones_like(position) * torch.tensor([0.12, -0.2])[None, :, None, None]
    weights = (position[:, 0] + 1.1).square()
    matrix, translation = weighted_affine(flow, weights)
    torch.testing.assert_close(matrix[0], torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(translation[0], torch.tensor([0.12, -0.2], dtype=torch.float64))


def test_zero_support_has_explicit_identity_fallback():
    flow = torch.randn(2, 2, 32, 32)
    matrix, translation = weighted_affine(flow, torch.zeros(2, 32, 32))
    assert torch.equal(matrix, torch.eye(2)[None].expand(2, -1, -1))
    assert torch.equal(translation, torch.zeros_like(translation))


def test_ratio_converts_to_current_ttc_with_correct_signed_clock():
    current = torch.tensor([0.2, 1.0, 4.0, -2.0], dtype=torch.float64)
    dt = torch.tensor([0.01, 0.05, 0.1, 0.05], dtype=torch.float64)
    ratio = torch.log1p(dt / current)
    actual = phase_to_ttc(relative_scale_phase(ratio, dt))
    torch.testing.assert_close(actual, current, atol=1e-10, rtol=1e-10)
    later = current - dt
    second = phase_to_ttc(relative_scale_phase(torch.log1p(dt / later), dt))
    torch.testing.assert_close(second, later, atol=1e-10, rtol=1e-10)
    assert not torch.equal(ratio, torch.log1p(dt / later))


def test_raw_phase_retains_finite_gradient_beyond_emitted_saturation():
    inverse = torch.tensor(
        [-20.0, 0.0, 9.8, 9.9, 10.0, 20.0], dtype=torch.float64, requires_grad=True
    )
    dt = torch.full_like(inverse, 0.01)
    raw = relative_scale_phase(torch.log1p(dt * inverse), dt)
    raw.sum().backward()
    assert torch.isfinite(raw).all() and torch.isfinite(inverse.grad).all()
    assert (inverse.grad > 0).all()
    torch.testing.assert_close(
        inverse.grad[-3:], torch.full((3,), 10.0, dtype=torch.float64), atol=1e-9, rtol=0
    )
    assert torch.isfinite(phase_to_ttc(raw)).all()


def test_zero_expansion_emits_native_positive_sixty_seconds():
    raw = relative_scale_phase(torch.zeros(3), torch.tensor([0.01, 0.05, 0.1]))
    torch.testing.assert_close(phase_to_ttc(raw), torch.full((3,), 60.0))


def test_common_tower_initialization_matches_free_readout_and_parent_is_private():
    parent = CausalScaleTTC(CausalScaleTTCConfig())
    torch.manual_seed(7)
    free = PairMeasurement.from_native(parent)
    torch.manual_seed(7)
    geometry = GeometricPairMeasurement(parent)
    for name, value in free.spatial.state_dict().items():
        assert torch.equal(value, geometry.spatial.state_dict()[name])
    for original, copied in zip(
        parent.encoder.features.parameters(), geometry.encoder.parameters(), strict=True
    ):
        assert torch.equal(original, copied)
        assert original.data_ptr() != copied.data_ptr()


def test_actual_native_encoder_receives_finite_nonzero_gradients_without_optimizer_updates():
    torch.manual_seed(19)
    parent = CausalScaleTTC(CausalScaleTTCConfig())
    model = GeometricPairMeasurement(parent).train()
    inputs = torch.randn(2, 2, 12, 128, 128)
    output = model(inputs, torch.tensor([0.05, 0.1]))
    assert output["affine_matrices"].shape == (2, 4, 2, 2)
    assert torch.isfinite(output["raw_phase"]).all()
    output["raw_phase"].sum().backward()
    for parameter in model.encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in parent.parameters())
