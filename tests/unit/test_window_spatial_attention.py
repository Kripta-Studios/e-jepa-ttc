from __future__ import annotations

import torch

from e_jepa_ttc.models.highres_factorized import WindowSpatialAttention


def test_window_attention_is_bidirectional_inside_a_frame() -> None:
    torch.manual_seed(17)
    module = WindowSpatialAttention(8, heads=2, window_size=2).eval()
    values = torch.zeros(1, 1, 2, 2, 8)
    changed = values.clone()
    changed[:, :, 0, 1, 0] = 3.0
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    with torch.inference_mode():
        reference = module(values, valid)
        result = module(changed, valid)
    assert not torch.allclose(reference[:, :, 0, 0], result[:, :, 0, 0])


def test_shifted_windows_transmit_information_across_a_regular_window_boundary() -> None:
    torch.manual_seed(23)
    regular = WindowSpatialAttention(8, heads=2, window_size=2).eval()
    shifted = WindowSpatialAttention(8, heads=2, window_size=2, shift_size=1).eval()
    values = torch.zeros(1, 1, 2, 4, 8)
    changed = values.clone()
    changed[:, :, 0, 0, 0] = 4.0
    valid = torch.ones(1, 1, 2, 4, dtype=torch.bool)

    with torch.inference_mode():
        reference = shifted(regular(values, valid), valid)
        result = shifted(regular(changed, valid), valid)
    assert not torch.allclose(reference[:, :, 0, 2], result[:, :, 0, 2])


def test_padding_mask_neither_contributes_to_attention_nor_output() -> None:
    torch.manual_seed(29)
    module = WindowSpatialAttention(8, heads=2, window_size=2).eval()
    values = torch.randn(1, 1, 2, 2, 8)
    changed = values.clone()
    changed[:, :, 1, 1] = 1000.0
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    valid[:, :, 1, 1] = False

    with torch.inference_mode():
        reference = module(values, valid)
        result = module(changed, valid)
    assert torch.allclose(reference[valid], result[valid], atol=1e-5)
    assert torch.equal(result[~valid], torch.zeros_like(result[~valid]))
