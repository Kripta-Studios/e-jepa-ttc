"""Tests for physics-constrained FlowMimic event generation."""

import torch

from e_jepa_ttc.representations.flowmimic import (
    generate_physical_event_approach_batch,
    physical_approach_scale,
)


def test_physical_approach_scale_matches_pinhole_solution() -> None:
    ttc = torch.tensor([2.0])
    delta = torch.tensor([0.0, 1.0])
    scale = physical_approach_scale(ttc[:, None], delta[None])
    torch.testing.assert_close(scale, torch.tensor([[1.0, 2.0]]))


def test_flowmimic_renders_events_before_voxelization() -> None:
    torch.manual_seed(4)
    batch = generate_physical_event_approach_batch(
        batch_size=3,
        output_channels=21,
        height=32,
        width=48,
        bins=5,
        horizons_ms=(50, 100, 250),
        context_ms=100.0,
        device=torch.device("cpu"),
        metadata_channels=True,
    )

    assert batch.context.shape == (3, 21, 32, 48)
    assert batch.future.shape == (3, 3, 21, 32, 48)
    assert batch.inverse_ttc_at_context_end.shape == (3,)
    assert torch.all(batch.context[:, :10].sum(dim=(1, 2, 3)) > 0.0)
    assert torch.all(batch.future[:, :, :10].sum(dim=(2, 3, 4)) > 0.0)
    assert torch.all(batch.context[:, 12:] == 0.0)
    assert torch.all(batch.future[:, :, 12:] == 0.0)
    assert torch.all(torch.isfinite(batch.context))


def test_flowmimic_generation_is_seed_deterministic() -> None:
    kwargs = {
        "batch_size": 2,
        "output_channels": 10,
        "height": 24,
        "width": 32,
        "bins": 5,
        "horizons_ms": (50, 100),
        "context_ms": 100.0,
        "device": torch.device("cpu"),
    }
    torch.manual_seed(9)
    first = generate_physical_event_approach_batch(**kwargs)
    torch.manual_seed(9)
    second = generate_physical_event_approach_batch(**kwargs)
    torch.testing.assert_close(first.context, second.context)
    torch.testing.assert_close(first.future, second.future)
    torch.testing.assert_close(
        first.inverse_ttc_at_context_end,
        second.inverse_ttc_at_context_end,
    )


def test_flowmimic_rejects_windows_that_cross_collision() -> None:
    try:
        generate_physical_event_approach_batch(
            batch_size=1,
            output_channels=10,
            height=24,
            width=32,
            bins=5,
            horizons_ms=(500,),
            context_ms=100.0,
            device=torch.device("cpu"),
            minimum_ttc_s=0.2,
            maximum_ttc_s=0.7,
        )
    except ValueError as error:
        assert "margin" in str(error)
    else:
        raise AssertionError("Unsafe FlowMimic time range was accepted.")
