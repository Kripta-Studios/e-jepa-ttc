"""Factorized Patch Policy spatial/temporal mixer."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.models.block_causal_transformer import BlockCausalTransformer
from e_jepa_ttc.models.spatial_patch_mixer import SpatialPatchMixer
from e_jepa_ttc.models.temporal_kda import KDALayoutMetadata, KimiDeltaAttention


class HybridSpatiotemporalMixer(nn.Module):
    """Select the reference block-causal or linear-memory temporal arm."""

    def __init__(
        self,
        dim: int,
        *,
        mode: str = "block_causal",
        heads: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if mode not in {"block_causal", "object_kda", "aligned_patch_kda"}:
            raise ValueError(f"Unsupported temporal mixer mode: {mode}")
        self.mode = mode
        self.temporal: nn.Module
        self.kda_layers = nn.ModuleList()
        self.global_refresh = nn.ModuleList()
        if mode == "block_causal":
            # Build the common temporal block first.  Under the same seed its
            # initialization is identical to the global matched control; the
            # dense arm then adds only the spatial Patch Policy mixer.
            self.temporal = BlockCausalTransformer(dim, heads=heads, depth=depth)
        else:
            self.temporal = nn.Identity()
            self.kda_layers = nn.ModuleList(
                KimiDeltaAttention(dim, heads=heads) for _ in range(depth)
            )
            if mode == "aligned_patch_kda":
                self.global_refresh = nn.ModuleList(
                    BlockCausalTransformer(dim, heads=heads, depth=1) for _ in range(depth // 2)
                )
        self.spatial = SpatialPatchMixer(dim, heads=heads)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Always resolve bidirectional spatial interaction before temporal mixing."""

        value = self.spatial(tokens)
        if self.mode == "block_causal":
            return self.temporal(value)
        refresh_index = 0
        metadata = KDALayoutMetadata(
            batch_size=value.shape[0],
            temporal_steps=value.shape[1],
            patch_count=value.shape[2],
            embedding_dim=value.shape[3],
        )
        for index, layer in enumerate(self.kda_layers, start=1):
            value = layer(value, metadata=metadata)
            if self.mode == "aligned_patch_kda" and index % 2 == 0:
                value = self.global_refresh[refresh_index](value)
                refresh_index += 1
        return value


__all__ = ["HybridSpatiotemporalMixer"]
