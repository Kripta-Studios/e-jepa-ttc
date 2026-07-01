"""Token-transformer encoder for event voxel grids."""

from __future__ import annotations

import math

import torch
from torch import nn


def _sincos_2d_position_embedding(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build deterministic 2D sine-cosine position embeddings."""

    if dim % 4 != 0:
        msg = "Token embedding dimension must be divisible by 4 for 2D sin-cos positions."
        raise ValueError(msg)
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    pieces = [
        torch.sin(grid_x.flatten()[:, None] * omega[None, :] * math.pi),
        torch.cos(grid_x.flatten()[:, None] * omega[None, :] * math.pi),
        torch.sin(grid_y.flatten()[:, None] * omega[None, :] * math.pi),
        torch.cos(grid_y.flatten()[:, None] * omega[None, :] * math.pi),
    ]
    return torch.cat(pieces, dim=1).to(dtype=dtype)


class EventTokenTransformerEncoder(nn.Module):
    """ViT-style encoder adapted to compact event voxel grids."""

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 192,
        patch_size: int = 8,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            msg = "embed_dim must be divisible by 4."
            raise ValueError(msg)
        if embed_dim % num_heads != 0:
            msg = "embed_dim must be divisible by num_heads."
            raise ValueError(msg)
        self.output_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        grid_h, grid_w = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.patch_norm(tokens)
        pos = _sincos_2d_position_embedding(
            grid_h,
            grid_w,
            self.output_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + pos[None, :, :]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into dense spatial tokens."""

        tokens = self._patch_tokens(x)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.final_norm(tokens)

    def forward_intermediate_tokens(
        self,
        x: torch.Tensor,
        layer_indices: tuple[int, ...],
    ) -> list[torch.Tensor]:
        """Encode a batch and return selected intermediate dense token layers."""

        if not layer_indices:
            return [self.forward_tokens(x)]
        depth = len(self.layers)
        if any(layer_idx < 0 or layer_idx >= depth for layer_idx in layer_indices):
            msg = f"layer_indices must be in [0, {depth - 1}], got {layer_indices}."
            raise ValueError(msg)
        selected = set(layer_indices)
        tokens = self._patch_tokens(x)
        outputs: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            tokens = layer(tokens)
            if layer_idx in selected:
                outputs.append(self.final_norm(tokens))
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into pooled latent vectors."""

        return self.forward_tokens(x).mean(dim=1)


class EventTokenTransformerRegressor(nn.Module):
    """Transformer token encoder with a log-TTC regression head."""

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 192,
        patch_size: int = 8,
        depth: int = 4,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = EventTokenTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.encoder.output_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log-TTC for a batch of voxel grids."""

        return self.head(self.encoder(x)).squeeze(-1)
