from __future__ import annotations

import torch

from e_jepa_ttc.models.highres_factorized import WindowSpatialAttention


def test_shifted_window_path_transmits_a_moving_patch_without_future_labels() -> None:
    torch.manual_seed(61)
    regular = WindowSpatialAttention(8, heads=2, window_size=2).eval()
    shifted = WindowSpatialAttention(8, heads=2, window_size=2, shift_size=1).eval()
    first = torch.zeros(1, 1, 2, 4, 8)
    second = first.clone()
    first[:, :, 0, 0, 0] = 3.0
    second[:, :, 0, 1, 0] = 3.0
    valid = torch.ones(1, 1, 2, 4, dtype=torch.bool)
    with torch.inference_mode():
        first_features = shifted(regular(first, valid), valid)
        second_features = shifted(regular(second, valid), valid)
    assert not torch.allclose(first_features, second_features)
