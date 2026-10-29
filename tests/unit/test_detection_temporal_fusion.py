from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.detection.temporal_fusion import (
    CausalTemporalFeatureFusion,
    TemporalFusionSpec,
)


def _inputs(
    channels: tuple[int, ...] = (4, 6), batch: int = 2
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    current = [torch.randn(batch, value, 5, 7) for value in channels]
    history = [torch.randn(batch, 2, value, 5, 7) for value in channels]
    ages = torch.tensor([[200_000.0, 100_000.0], [0.0, 100_000.0]])
    mask = torch.tensor([[True, True], [False, True]])
    return current, history, ages, mask


def _model(channels: tuple[int, ...] = (4, 6)) -> CausalTemporalFeatureFusion:
    return CausalTemporalFeatureFusion(
        TemporalFusionSpec(channels=channels, bottleneck_channels=3)
    )


def test_identity_initialization_and_single_frame_exactness() -> None:
    torch.manual_seed(7)
    model = _model()
    current, history, ages, mask = _inputs()
    fused = model(current, history, ages, mask)
    assert all(
        torch.equal(actual, expected) for actual, expected in zip(fused, current, strict=True)
    )

    with torch.no_grad():
        for projection in model.output_projections():
            projection.weight.fill_(0.2)
            assert projection.bias is not None
            projection.bias.fill_(0.1)
    single = model(current, history_limit=0)
    assert all(
        torch.equal(actual, expected) for actual, expected in zip(single, current, strict=True)
    )


def test_detached_history_and_current_and_fusion_gradients() -> None:
    torch.manual_seed(13)
    model = _model((4,))
    current, history, ages, mask = _inputs((4,), batch=1)
    current[0].requires_grad_(True)
    history[0].requires_grad_(True)
    result = model(current, history, ages[:1], mask[:1])[0]
    result.square().sum().backward()

    assert current[0].grad is not None
    assert torch.count_nonzero(current[0].grad) > 0
    assert history[0].grad is None
    output_projection = model.output_projections()[0]
    assert output_projection.weight.grad is not None
    assert torch.count_nonzero(output_projection.weight.grad) > 0


def test_missing_history_is_exact_identity_per_batch_item_after_training() -> None:
    model = _model((4,))
    current, history, ages, mask = _inputs((4,))
    mask[1] = False
    ages[1] = 0
    with torch.no_grad():
        projection = model.output_projections()[0]
        projection.weight.fill_(0.1)
        assert projection.bias is not None
        projection.bias.fill_(0.2)
    result = model(current, history, ages, mask)[0]
    assert not torch.equal(result[0], current[0][0])
    assert torch.equal(result[1], current[0][1])


def test_age_order_and_repeated_current_capacity_change_learned_output() -> None:
    torch.manual_seed(23)
    model = _model((4,))
    current, history, ages, mask = _inputs((4,), batch=1)
    with torch.no_grad():
        model.output_projections()[0].weight.fill_(0.1)
    original = model(current, history, ages[:1], mask[:1])[0]
    swapped = model(
        current,
        [history[0].flip(1)],
        ages[:1].flip(1),
        mask[:1].flip(1),
    )[0]
    repeated = model(
        current,
        [current[0].unsqueeze(1).repeat(1, 2, 1, 1, 1)],
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )[0]
    assert not torch.allclose(original, swapped)
    assert not torch.allclose(repeated, current[0])


def test_one_history_uses_newer_slot() -> None:
    model = _model((4,))
    current, history, ages, mask = _inputs((4,), batch=1)
    mask[:, 0] = False
    ages[:, 0] = 0
    assert model(current, history, ages[:1], mask[:1], history_limit=1)[0].shape == current[0].shape
    mask[:, 0] = True
    ages[:, 0] = 200_000
    with pytest.raises(ValueError, match="newer second slot"):
        model(current, history, ages[:1], mask[:1], history_limit=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nan_age", "finite"),
        ("negative_age", "nonnegative"),
        ("missing_nonzero_age", "zero age"),
        ("float_mask", "boolean"),
        ("wrong_history_shape", "shape"),
    ],
)
def test_invalid_temporal_inputs_fail_closed(mutation: str, message: str) -> None:
    model = _model((4,))
    current, history, ages, mask = _inputs((4,))
    mask.fill_(False)
    ages.zero_()
    if mutation == "nan_age":
        ages.fill_(float("nan"))
    elif mutation == "negative_age":
        ages.fill_(-1)
    elif mutation == "missing_nonzero_age":
        ages.fill_(1)
    elif mutation == "float_mask":
        mask = mask.float()
    else:
        history[0] = history[0][:, :, :, :-1]
    with pytest.raises(ValueError, match=message):
        model(current, history, ages, mask)


def test_spec_and_parameter_receipt_are_strict() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        TemporalFusionSpec(history_slots=3)
    model = _model()
    assert model.parameter_count() == sum(parameter.numel() for parameter in model.parameters())
    assert model.parameter_count() > 0


def test_current_scales_require_common_dtype() -> None:
    model = _model()
    current, history, ages, mask = _inputs()
    current[1] = current[1].double()
    history[1] = history[1].double()
    with pytest.raises(ValueError, match="device and dtype"):
        model(current, history, ages, mask)
