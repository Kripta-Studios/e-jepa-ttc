from __future__ import annotations

import torch

from e_jepa_ttc.training.jepa import _mask_context


def test_tubelet_mask_only_erases_event_channels_and_preserves_auxiliary_channels() -> None:
    torch.manual_seed(5)
    values = torch.ones(2, 8, 12, 12)
    values[:, 4:] = 7.0
    masked = _mask_context(
        values,
        mask_ratio=0.5,
        block_count=3,
        mask_mode="tubelet",
        event_bins=2,
    )

    assert torch.equal(masked[:, 4:], values[:, 4:])
    assert bool((masked[:, :4] == 0).any())
    assert torch.equal(
        _mask_context(values, mask_ratio=0.0, block_count=3, mask_mode="tubelet", event_bins=2),
        values,
    )


def test_spatial_mask_has_no_future_or_label_argument() -> None:
    values = torch.ones(1, 4, 8, 8)
    masked = _mask_context(
        values,
        mask_ratio=0.25,
        block_count=2,
        mask_mode="spatial",
        event_bins=2,
    )
    assert masked.shape == values.shape
    assert bool((masked == 0).any())
