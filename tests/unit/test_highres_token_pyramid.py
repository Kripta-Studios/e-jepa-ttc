from __future__ import annotations

import torch

from e_jepa_ttc.models.highres_factorized import space_to_depth_2x2
from e_jepa_ttc.models.highres_token_pyramid import (
    HighResolutionTokenPyramid,
    SpaceToDepthPatchMerge,
)


def test_space_to_depth_preserves_checkerboard_before_projection_and_is_invertible() -> None:
    values = torch.tensor([[[[[1.0], [2.0]], [[3.0], [4.0]]]]])
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    merged, merged_mask = space_to_depth_2x2(values, valid)
    reconstructed = (
        merged.reshape(1, 1, 1, 1, 2, 2, 1).permute(0, 1, 2, 4, 3, 5, 6).reshape_as(values)
    )
    assert torch.equal(reconstructed, values)
    assert bool(merged_mask.item())


def test_pyramid_merge_and_valid_mask_have_exact_token_counts() -> None:
    torch.manual_seed(31)
    values = torch.randn(2, 3, 4, 4, 8)
    valid = torch.ones(2, 3, 4, 4, dtype=torch.bool)
    valid[:, :, -1, -1] = False
    merge = SpaceToDepthPatchMerge(8, 16).eval()
    pyramid = HighResolutionTokenPyramid(
        8,
        output_dim=16,
        heads=2,
        window_size=2,
        merge=True,
        shift_size=1,
    ).eval()

    with torch.inference_mode():
        merged, merged_mask = merge(values, valid)
        output, output_mask = pyramid(values, valid)
    assert merged.shape == (2, 3, 2, 2, 16)
    assert output.shape == merged.shape
    assert merged_mask.shape == output_mask.shape == (2, 3, 2, 2)
    # The validity contract is OR over the four children: one valid child is
    # enough for a merged patch to remain usable.
    assert int(output_mask.sum()) == 24


def test_pyramid_exposes_dense_pre_and_post_merge_taps() -> None:
    torch.manual_seed(41)
    values = torch.randn(1, 2, 3, 5, 8)
    valid = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    pyramid = HighResolutionTokenPyramid(
        8,
        output_dim=16,
        heads=2,
        window_size=2,
        spatial_depth=2,
        merge=True,
    ).eval()

    with torch.inference_mode():
        output, output_mask, taps = pyramid.forward_with_taps(values, valid)

    assert taps["pre_merge_tokens"].shape == (1, 2, 3, 5, 8)
    assert taps["post_merge_tokens"].shape == output.shape == (1, 2, 2, 3, 16)
    assert taps["pre_merge_mask"].shape == (1, 2, 3, 5)
    assert taps["post_merge_mask"].shape == output_mask.shape == (1, 2, 2, 3)
    assert pyramid.spatial[0].shift_size == 0
    assert pyramid.spatial[1].shift_size == 1
