from __future__ import annotations

import pytest
import torch
from torch import nn

from e_jepa_ttc.detection.event_detector import EventTemporalDetectorAdapter
from e_jepa_ttc.detection.event_fusion import (
    CausalEventFeatureFusion,
    EventFusionObservables,
    EventFusionSpec,
)
from e_jepa_ttc.detection.temporal_detector import (
    ProjectedFeatureBackend,
    ProjectedFeatureState,
    TemporalBatchContext,
    TemporalDetectorAdapter,
    TemporalFrameIdentity,
)
from e_jepa_ttc.detection.temporal_fusion import CausalTemporalFeatureFusion


class Backend(ProjectedFeatureBackend):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 256, 1)

    def extract_projected_features(self, images):
        projected = self.projection(images)
        return ProjectedFeatureState(
            tuple(
                nn.functional.adaptive_avg_pool2d(projected, shape)
                for shape in [(8, 12), (4, 6), (2, 3), (1, 2)]
            ),
            "current-context",
        )

    def decode_projected_features(
        self, state, features, calibs, targets, image_sizes, dn_args=None
    ):
        assert state.decoder_context == "current-context"
        return {"pred_logits": sum(feature.mean((2, 3)) for feature in features)}


def setup_model():
    torch.manual_seed(7)
    backend = Backend()
    temporal = CausalTemporalFeatureFusion()
    event = CausalEventFeatureFusion(
        EventFusionSpec(input_height=32, input_width=48, encoder_channels=(4, 8, 8, 8))
    )
    model = EventTemporalDetectorAdapter(
        backend, temporal, event, producer_version="test", maximum_history_age_us=500_000
    )
    original = TemporalDetectorAdapter(
        backend, temporal, producer_version="test", maximum_history_age_us=500_000
    )
    context = TemporalBatchContext(
        "test", "current_only", (TemporalFrameIdentity("s", "s:current"),), ((None, None),)
    )
    images = torch.randn(1, 3, 16, 24, requires_grad=True)
    events = dict(
        event_control="F01",
        event_tokens=("s:current",),
        event_voxel=torch.randn(1, 10, 32, 48),
        event_mask=torch.tensor([True]),
        event_observables=EventFusionObservables(torch.tensor([20.0]), torch.tensor([50_000.0])),
    )
    return model, original, context, images, events


def test_event_zero_initialization_preserves_full_decoder_and_learns_residual() -> None:
    model, original, context, images, events = setup_model()
    args = (images, torch.empty(0), None, torch.empty(0))
    expected = original(*args, temporal_context=context)["pred_logits"]
    actual = model(*args, temporal_context=context, **events)["pred_logits"]
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    for projection in model.event_fusion.output_projections():
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        for projection in model.event_fusion.output_projections():
            projection.bias.fill_(0.1)
    changed = model(*args, temporal_context=context, **events)["pred_logits"]
    assert not torch.equal(changed, expected)
    restored = model(*args, temporal_context=context, event_control="F00")["pred_logits"]
    assert torch.equal(restored, expected)  # No event request can leak into the next call.


def test_event_identity_and_sensor_control_fail_closed() -> None:
    model, _, context, images, events = setup_model()
    args = (images, torch.empty(0), None, torch.empty(0))
    with pytest.raises(ValueError, match="identities"):
        model(*args, temporal_context=context, **(events | {"event_tokens": ("other:f",)}))
    with pytest.raises(ValueError, match="history setting"):
        model(*args, temporal_context=context, **(events | {"event_control": "F11"}))
    with pytest.raises(ValueError, match="explicit tensors"):
        model(*args, temporal_context=context, **(events | {"event_voxel": None}))


def test_combined_fusion_keeps_history_detached() -> None:
    model, _, _, images, events = setup_model()
    context = TemporalBatchContext(
        "test",
        "correct_past",
        (TemporalFrameIdentity("s", "s:current"),),
        ((TemporalFrameIdentity("s", "s:p0"), TemporalFrameIdentity("s", "s:p1")),),
    )
    history = torch.randn(1, 2, 3, 16, 24, requires_grad=True)
    result = model(
        images,
        torch.empty(0),
        None,
        torch.empty(0),
        temporal_context=context,
        history_images=history,
        history_ages_us=torch.tensor([[100000.0, 50000.0]]),
        history_mask=torch.tensor([[True, True]]),
        history_limit=2,
        **(events | {"event_control": "F11"}),
    )
    result["pred_logits"].sum().backward()
    assert history.grad is None
    assert images.grad is not None and torch.isfinite(images.grad).all()
