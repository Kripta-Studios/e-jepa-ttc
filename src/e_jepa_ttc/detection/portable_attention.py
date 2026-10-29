"""Differentiable PyTorch deformable attention for the portable detector variant.

Adapted from the Apache-2.0 reference implementation in Deformable DETR,
Copyright (c) 2020 SenseTime, and the pinned MonoDGP reference at
aa059a18214aebf644510e7f0793971b403f9d14. This backend uses PyTorch grid_sample;
it does not load or claim parity with a compiled external CUDA extension.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def deformable_attention_pytorch(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Sample normalized (x,y) locations with zero padding, summing level weights.

    Shapes: value [B,S,H,D], spatial_shapes [L,2] in height/width order,
    sampling_locations [B,Q,H,L,P,2], weights [B,Q,H,L,P]. CUDA grid_sample
    backward can be nondeterministic; seeded runs are not bitwise guarantees.
    """
    if value.ndim != 4 or spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
        raise ValueError("Expected value [B,S,H,D] and spatial_shapes [L,2].")
    if sampling_locations.ndim != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError("Expected sampling_locations [B,Q,H,L,P,2].")
    batch, length, heads, channels = value.shape
    shapes = [(int(height), int(width)) for height, width in spatial_shapes.tolist()]
    sizes = [height * width for height, width in shapes]
    if not shapes or any(height <= 0 or width <= 0 for height, width in shapes):
        raise ValueError("Spatial dimensions must be positive.")
    if sum(sizes) != length:
        raise ValueError("Spatial shapes do not cover the flattened value sequence.")
    if sampling_locations.shape[:1] != (batch,) or sampling_locations.shape[2] != heads:
        raise ValueError("Sampling batch/head dimensions disagree with values.")
    if sampling_locations.shape[3] != len(shapes):
        raise ValueError("Sampling level count disagrees with spatial shapes.")
    if attention_weights.shape != sampling_locations.shape[:-1]:
        raise ValueError("Attention weights must match sampling locations without xy.")
    if not torch.isfinite(sampling_locations).all().item():
        raise ValueError("Sampling locations must be finite before grid_sample.")
    queries = sampling_locations.shape[1]
    points = sampling_locations.shape[4]
    grids = 2 * sampling_locations - 1
    output: Tensor | None = None
    for level, (level_value, (height, width)) in enumerate(
        zip(value.split(sizes, dim=1), shapes, strict=True)
    ):
        feature = (
            level_value.flatten(2).transpose(1, 2).reshape(batch * heads, channels, height, width)
        )
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        samples = functional.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        weights = (
            attention_weights[:, :, :, level]
            .transpose(1, 2)
            .reshape(batch * heads, 1, queries, points)
        )
        contribution = (samples * weights).sum(-1)
        output = contribution if output is None else output + contribution
    assert output is not None
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


class PortableMSDeformAttnFunction:
    """API adapter preserving native PyTorch autograd through ordinary operations."""

    @staticmethod
    def apply(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ) -> Tensor:
        """Accept the upstream call signature without using an autograd.Function."""
        del im2col_step  # A custom-kernel batching hint, irrelevant to grid_sample.
        sizes = spatial_shapes.prod(1)
        expected = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
        if not torch.equal(level_start_index, expected):
            raise ValueError("level_start_index must describe contiguous feature levels.")
        return deformable_attention_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
