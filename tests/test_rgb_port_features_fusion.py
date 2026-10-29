"""Zero-update contracts for RGB-PORT features and controlled fusion."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_SHA256,
    PAIR_E_SCHEMA_SHA256,
    PAIR_R_SCHEMA_SHA256,
    RGB_PHASE17_SHA256,
    ProducerObservation,
    event_phase17,
    event_statistics_from_channels12,
    pair133,
    raw_rgb_statistics,
    rgb_phase17,
)
from e_jepa_ttc.rgb_port.fusion import DualClockFusion, MissingRGBFallback, fusion_loss


def _stream(batch: int = 2, length: int = 3):
    features = torch.randn(batch, length, 17)
    timing = torch.randn(batch, length, 4)
    valid = torch.tensor([[False, True, True], [True, True, True]])
    return features, timing, valid


def test_pair133_has_exact_features_and_distinct_modality_hashes() -> None:
    token = torch.randn(4, 128)
    result = pair133(token, torch.full((4,), 0.1), torch.rand(4, 2))
    assert result.shape == (4, 133)
    assert PAIR_E_SCHEMA_SHA256 != PAIR_R_SCHEMA_SHA256
    assert RGB_PHASE17_SHA256 != EVENT_PHASE17_SHA256


def test_producer_observation_uses_current_then_minimum_support() -> None:
    output = {
        "token128": torch.zeros(2, 128),
        "prediction_ttc": torch.tensor([0.05, -2.0]),
        "point_phase": torch.tensor([8.0, -0.1]),
        "flow": torch.ones(2),
        "margin": torch.ones(2),
        "log_variance": torch.zeros(2),
        "support": torch.tensor([[0.8, 0.2], [0.1, 0.7]]),
        "known": torch.ones(2, 2, dtype=torch.bool),
    }
    observation = ProducerObservation.from_output(output)
    features = observation.pair_input(torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(features[:, -2:], torch.tensor([[0.2, 0.2], [0.7, 0.1]]))


def test_rgb_phase17_uses_raw_unit_rgb_and_exact_disagreement_order() -> None:
    rgb = torch.zeros(2, 3, 4, 4)
    rgb[:, :, :, 2:] = 1
    stats = raw_rgb_statistics(rgb)
    assert stats.shape == (2, 2) and (stats[:, 1] > 0).all()
    phase = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    result = rgb_phase17(rgb, torch.zeros(2, 3), torch.zeros(2, 3), phase)
    assert result.shape == (2, 17)
    torch.testing.assert_close(result[:, -3:], result[:, -6:-3].abs())
    try:
        raw_rgb_statistics(rgb - 2)
    except ValueError as error:
        assert "raw [0,1]" in str(error)
    else:
        raise AssertionError("ImageNet-like negative RGB entered raw statistics")


def test_event_phase17_consumes_frozen_phases_without_ttc_reconversion() -> None:
    phase = torch.tensor([[0.01, 0.02, 0.03], [-0.1, -0.2, -0.3]])
    result = event_phase17(torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2, 3), phase)
    torch.testing.assert_close(result[:, 8:11], phase, atol=0, rtol=0)


def test_event_statistics_are_canonical_channel10_count_channel11_rate_means() -> None:
    inputs = torch.zeros(2, 3, 12, 2, 4)
    inputs[..., 10, :, :] = torch.tensor([1.0, 3.0]).reshape(2, 1, 1, 1)
    inputs[..., 11, :, :] = torch.tensor([2.0, 6.0]).reshape(2, 1, 1, 1)
    result = event_statistics_from_channels12(inputs)
    assert result.shape == (2, 3, 2)
    torch.testing.assert_close(result[0], torch.tensor([[1.0, 2.0]]).expand(3, -1))
    torch.testing.assert_close(result[1], torch.tensor([[3.0, 6.0]]).expand(3, -1))


def test_f_zero_is_invariant_to_all_rgb_predictive_features() -> None:
    torch.manual_seed(7)
    model = DualClockFusion("F_ZERO").eval()
    event, event_timing, event_valid = _stream()
    rgb, rgb_timing, rgb_valid = _stream()
    expert = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    with torch.no_grad():
        left = model(event, event_timing, event_valid, expert, rgb, rgb_timing, rgb_valid)
        right = model(event, event_timing, event_valid, expert, rgb + 1000, rgb_timing, rgb_valid)
    for key in ("point_phase", "raw_location", "q10", "q90", "rgb_hidden"):
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_fusion_loss_has_no_expert_cost_term() -> None:
    model = DualClockFusion("F_TRUE")
    event, event_timing, event_valid = _stream()
    rgb, rgb_timing, rgb_valid = _stream()
    expert = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    output = model(event, event_timing, event_valid, expert, rgb, rgb_timing, rgb_valid)
    loss = fusion_loss(output, torch.tensor([0.2, -0.2]), torch.tensor([0.4, 0.6]), 2)
    assert loss.ndim == 0 and torch.isfinite(loss)


class _EventStub(nn.Module):
    def forward(self, features, timing, valid, experts):
        del timing, valid, experts
        value = features[:, -1, 0]
        return {"point_phase": value, "raw_location": value, "q10": value, "q90": value}


class _NeverFusion(DualClockFusion):
    def forward(self, *args, **kwargs):
        raise AssertionError("fusion/RGB path was invoked while RGB unavailable")


def test_missing_rgb_returns_event_context_exactly_without_fusion() -> None:
    event, timing, valid = _stream()
    expert = torch.zeros(2, 3)
    expected = _EventStub()(event, timing, valid, expert)
    wrapper = MissingRGBFallback(_EventStub(), _NeverFusion("F_TRUE"))
    actual = wrapper(
        event,
        timing,
        valid,
        expert,
        rgb_available=torch.zeros(2, dtype=torch.bool),
    )
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


class _CountingFusion(DualClockFusion):
    calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        assert len(args[0]) == 1
        assert torch.isfinite(args[4]).all()
        return super().forward(*args, **kwargs)


def test_mixed_missing_rgb_slices_before_fusion_and_preserves_event_row() -> None:
    event, timing, valid = _stream()
    expert = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    rgb, rgb_timing, rgb_valid = _stream()
    rgb[0] = torch.nan
    rgb_timing[0] = torch.nan
    rgb_valid[0] = False
    fallback = MissingRGBFallback(_EventStub(), _CountingFusion("F_TRUE"))
    expected = _EventStub()(event, timing, valid, expert)
    actual = fallback(
        event,
        timing,
        valid,
        expert,
        rgb_available=torch.tensor([False, True]),
        rgb_features=rgb,
        rgb_timing=rgb_timing,
        rgb_valid=rgb_valid,
    )
    assert fallback.fusion.calls == 1
    for key in expected:
        torch.testing.assert_close(actual[key][0], expected[key][0], atol=0, rtol=0)
