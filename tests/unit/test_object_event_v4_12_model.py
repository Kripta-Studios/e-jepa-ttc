from types import SimpleNamespace

import torch
from torch import nn

from e_jepa_ttc.models.object_event_v4_12 import (
    ObjectEventTTCV412,
    ObjectEventV412Config,
    _spatial_directional_moments,
    _weighted_mean_std,
)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_abs_expansion=0.25)
        self.motion_config = SimpleNamespace(
            motion_refine_dim=8,
            field_size=16,
            maximum_abs_log_eta=0.30,
            confidence_floor=0.10,
            activity_floor=0.10,
            weight_epsilon=1.0e-5,
        )
        self.temporal_projection = nn.Conv2d(12, 8, kernel_size=1)
        self.field_head = nn.Conv2d(8 + 8, 2, kernel_size=1)

    def _foreground_and_features(self, events: torch.Tensor):
        batch = events.shape[0]
        maps = events[:, :, :2, :4, :4]
        foreground = torch.sigmoid(events[:, :, 0, :16, :16])
        activity = events[:, :, 1, :16, :16]
        logits = torch.logit(foreground.clamp(1.0e-4, 1.0 - 1.0e-4))
        return maps, logits, foreground, activity

    @staticmethod
    def _temporal_maps(maps: torch.Tensor) -> torch.Tensor:
        m0, m1, m2 = maps.unbind(dim=1)
        d01 = m1 - m0
        d12 = m2 - m1
        acc = d12 - d01
        return torch.cat((d01, d12, acc, d01.abs(), d12.abs(), acc.abs()), dim=1)

    @staticmethod
    def _activity_features(activity: torch.Tensor) -> torch.Tensor:
        a0, a1, a2 = activity.unbind(dim=1)
        d01 = a1 - a0
        d12 = a2 - a1
        return torch.stack((a0, a1, a2, d01, d12, d12 - d01), dim=1)


def test_weighted_statistics_and_moments_are_finite() -> None:
    features = torch.randn(3, 5, 8, 8)
    weights = torch.rand(3, 8, 8)
    mean, std = _weighted_mean_std(features, weights, epsilon=1.0e-6)
    moments = _spatial_directional_moments(features[:, :2], weights, epsilon=1.0e-6)
    assert mean.shape == (3, 5)
    assert std.shape == (3, 5)
    assert moments.shape == (3, 6)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.isfinite(moments).all()


def test_v412_forward_uses_external_magnitude_and_event_only_sign() -> None:
    torch.manual_seed(3)
    model = ObjectEventTTCV412(_FakeBackbone())
    events = torch.randn(4, 3, 2, 16, 16)
    magnitude = torch.tensor([0.01, -0.02, 0.03, -0.04])
    output = model(events, magnitude_expansion=magnitude)
    assert output.signed_expansion.shape == (4,)
    assert output.sign_logits.shape == (4,)
    assert output.descriptor.shape == (4, model.descriptor_dim)
    assert torch.allclose(output.signed_expansion.abs(), magnitude.abs(), atol=1.0e-6)
    assert torch.isfinite(output.negative_probability).all()


def test_v4121_sign_logits_are_exactly_odd_under_time_reversal() -> None:
    torch.manual_seed(11)
    model = ObjectEventTTCV412(_FakeBackbone()).train()
    events = torch.randn(5, 3, 2, 16, 16)
    forward = model(events)
    reverse = model(events.flip(1))
    assert torch.allclose(reverse.descriptor, -forward.descriptor, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(reverse.sign_logits, -forward.sign_logits, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(
        reverse.negative_probability,
        1.0 - forward.negative_probability,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_v4121_rejects_dropout_that_breaks_exact_odd_symmetry() -> None:
    try:
        ObjectEventV412Config(dropout=0.1)
    except ValueError as error:
        assert "dropout=0" in str(error)
    else:
        raise AssertionError("non-zero dropout should be rejected")
