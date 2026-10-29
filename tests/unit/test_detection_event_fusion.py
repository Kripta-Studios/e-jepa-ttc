from __future__ import annotations

import pytest
import torch
from torch import nn

from e_jepa_ttc.detection.event_fusion import (
    CausalEventFeatureFusion,
    EventFusionObservables,
    EventFusionSpec,
)


def small_spec() -> EventFusionSpec:
    return EventFusionSpec(input_height=32, input_width=48, encoder_channels=(4, 8, 8, 8))


def features(batch: int = 3) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.randn(batch, 256, height, width)
        for height, width in ((8, 12), (4, 6), (2, 3), (1, 2))
    )


def observables(
    counts: list[float], durations: list[float]
) -> EventFusionObservables:
    return EventFusionObservables(torch.tensor(counts), torch.tensor(durations))


def test_rgb_controls_and_missing_event_batch_are_exact_identity() -> None:
    module = CausalEventFeatureFusion(small_spec())
    current = features()
    for control in ("F00", "F10", "F01", "F11"):
        result = module(current, control=control)  # type: ignore[arg-type]
        assert all(output is source for output, source in zip(result, current, strict=True))


def test_zero_initialized_event_residual_is_exact_rgb_baseline() -> None:
    torch.manual_seed(7)
    module = CausalEventFeatureFusion(small_spec())
    current = features(batch=2)
    voxel = torch.randn(2, 10, 32, 48)
    result = module(
        current,
        control="F01",
        event_voxel=voxel,
        event_mask=torch.tensor([True, True]),
        observables=observables([100.0, 200.0], [50_000.0, 50_000.0]),
    )
    assert all(torch.equal(output, source) for output, source in zip(result, current, strict=True))
    assert all(
        not projection.weight.count_nonzero()
        for projection in module.output_projections()
    )


def test_valid_empty_and_missing_samples_fall_back_after_residual_is_enabled() -> None:
    module = CausalEventFeatureFusion(small_spec())
    for projection in module.output_projections():
        nn.init.zeros_(projection.weight)
        assert projection.bias is not None
        nn.init.ones_(projection.bias)
    current = features()
    result = module(
        current,
        control="F11",
        event_voxel=torch.zeros(3, 10, 32, 48),
        event_mask=torch.tensor([True, True, False]),
        observables=observables(
            [10.0, 0.0, 0.0],
            [50_000.0, 50_000.0, 0.0],
        ),
    )
    for output, source in zip(result, current, strict=True):
        assert torch.equal(output[0], source[0] + 1)
        assert torch.equal(output[1], source[1])
        assert torch.equal(output[2], source[2])


def test_mask_blocks_event_gradients_for_missing_sample() -> None:
    torch.manual_seed(13)
    module = CausalEventFeatureFusion(small_spec())
    for projection in module.output_projections():
        nn.init.constant_(projection.weight, 0.01)
        assert projection.bias is not None
        nn.init.zeros_(projection.bias)
    current = tuple(value.requires_grad_() for value in features(batch=2))
    voxel = torch.randn(2, 10, 32, 48, requires_grad=True)
    result = module(
        current,
        control="F01",
        event_voxel=voxel,
        event_mask=torch.tensor([True, False]),
        observables=observables([20.0, 0.0], [50_000.0, 0.0]),
    )
    torch.stack([output.sum() for output in result]).sum().backward()
    assert voxel.grad is not None
    assert voxel.grad[0].abs().sum() > 0
    assert not voxel.grad[1].count_nonzero()
    assert all(value.grad is not None for value in current)


def test_fixed_observables_are_label_free_detached_inputs() -> None:
    raw_count = torch.tensor([0.0, 99.0], requires_grad=True)
    duration = torch.tensor([50_000.0, 100_000.0], requires_grad=True)
    fixed = EventFusionObservables(raw_count, duration).fixed_gate_inputs()
    assert fixed.shape == (2, 2)
    assert not fixed.requires_grad
    assert torch.equal(fixed[:, 1], torch.tensor([0.5, 1.0]))


def test_ambiguous_or_invalid_event_inputs_fail_closed() -> None:
    module = CausalEventFeatureFusion(small_spec())
    current = features(batch=1)
    voxel = torch.zeros(1, 10, 32, 48)
    with pytest.raises(ValueError, match="must not receive"):
        module(current, control="F00", event_voxel=voxel)
    with pytest.raises(ValueError, match="together"):
        module(current, control="F01", event_voxel=voxel)
    with pytest.raises(ValueError, match="positive duration"):
        module(
            current,
            control="F01",
            event_voxel=voxel,
            event_mask=torch.tensor([True]),
            observables=observables([1.0], [0.0]),
        )
    with pytest.raises(ValueError, match="zero observables"):
        module(
            current,
            control="F01",
            event_voxel=voxel,
            event_mask=torch.tensor([False]),
            observables=observables([1.0], [50_000.0]),
        )


def test_default_contract_has_four_c256_scales_and_no_learned_normalizer() -> None:
    module = CausalEventFeatureFusion()
    assert module.spec.projected_channels == (256, 256, 256, 256)
    assert module.parameter_count() > 0
    assert not any(
        isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
        for layer in module.modules()
    )
