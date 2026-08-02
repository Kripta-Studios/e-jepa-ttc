"""Reusable high-resolution patch merge built on the audited factorized path."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from e_jepa_ttc.models.highres_factorized import (
    WindowSpatialAttention,
    space_to_depth_2x2,
)


class SpaceToDepthPatchMerge(nn.Module):
    """Losslessly rearrange valid 2x2 patches before learned projection."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim * 4, output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge ``[B,T,H,W,D]`` and propagate the OR validity mask."""

        if tokens.shape[-1] != self.input_dim:
            raise ValueError("tokens last dimension does not match input_dim.")
        merged, mask = space_to_depth_2x2(tokens, valid_patch_mask)
        return self.projection(merged), mask


class PyramidTaps(TypedDict):
    """Dense tensors exposed for localized JEPA losses and diagnostics."""

    input_tokens: torch.Tensor
    input_mask: torch.Tensor
    pre_merge_tokens: torch.Tensor
    pre_merge_mask: torch.Tensor
    post_merge_tokens: torch.Tensor
    post_merge_mask: torch.Tensor


class HighResolutionTokenPyramid(nn.Module):
    """Auditable local-spatial + optional 2x2 merge pyramid stage.

    ``forward`` keeps the compact two-output API used by existing callers.
    ``forward_with_taps`` additionally exposes the dense pre/post-merge values
    required by localized JEPA losses without flattening spatial patches into a
    temporal axis.
    """

    def __init__(
        self,
        dim: int,
        *,
        output_dim: int | None = None,
        heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        merge: bool = True,
        spatial_depth: int = 1,
    ) -> None:
        super().__init__()
        if spatial_depth <= 0:
            raise ValueError("spatial_depth must be positive.")
        alternating_shift = shift_size if shift_size > 0 else window_size // 2
        self.spatial = nn.ModuleList(
            WindowSpatialAttention(
                dim,
                heads=heads,
                window_size=window_size,
                # With multiple local blocks, alternate regular and shifted
                # windows by default.  An explicit shift_size overrides the
                # half-window amount while preserving the alternation.
                shift_size=(
                    shift_size
                    if spatial_depth == 1
                    else (alternating_shift if index % 2 and window_size > 1 else 0)
                ),
            )
            for index in range(spatial_depth)
        )
        self.merge = SpaceToDepthPatchMerge(dim, output_dim or dim) if merge else None

    def forward_with_taps(
        self,
        tokens: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PyramidTaps]:
        """Apply local windows and return dense pre/post-merge taps."""

        value = tokens
        for layer in self.spatial:
            value = layer(value, valid_patch_mask)
        mask = valid_patch_mask
        pre_merge_tokens = value
        pre_merge_mask = mask
        if self.merge is None:
            post_merge_tokens = value
            post_merge_mask = mask
        else:
            post_merge_tokens, post_merge_mask = self.merge(value, mask)
        taps: PyramidTaps = {
            "input_tokens": tokens,
            "input_mask": valid_patch_mask,
            "pre_merge_tokens": pre_merge_tokens,
            "pre_merge_mask": pre_merge_mask,
            "post_merge_tokens": post_merge_tokens,
            "post_merge_mask": post_merge_mask,
        }
        return post_merge_tokens, post_merge_mask, taps

    def forward(
        self,
        tokens: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional spatial windows and preserve padding semantics."""

        value, mask, _ = self.forward_with_taps(tokens, valid_patch_mask)
        return value, mask


__all__ = ["HighResolutionTokenPyramid", "PyramidTaps", "SpaceToDepthPatchMerge"]
