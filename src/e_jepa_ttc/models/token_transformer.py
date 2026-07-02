"""Token-transformer encoder for event voxel grids."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


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


def _sincos_1d_position_embedding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build deterministic 1D sine-cosine position embeddings."""

    if dim % 2 != 0:
        msg = "Token embedding dimension must be even for 1D sin-cos positions."
        raise ValueError(msg)
    position = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32)
    omega = torch.arange(dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 2 - 1, 1)))
    pieces = [
        torch.sin(position[:, None] * omega[None, :] * math.pi),
        torch.cos(position[:, None] * omega[None, :] * math.pi),
    ]
    return torch.cat(pieces, dim=1).to(dtype=dtype)


def _split_rotary_dims(head_dim: int, axis_count: int = 3) -> tuple[int, ...]:
    """Split a head dimension into even rotary sections for factorized axes."""

    if head_dim % 2 != 0:
        msg = "Rotary attention requires an even attention head dimension."
        raise ValueError(msg)
    if head_dim < axis_count * 2:
        msg = (
            f"Rotary attention requires at least {axis_count * 2} head dimensions "
            f"for {axis_count} axes, got {head_dim}."
        )
        raise ValueError(msg)
    pair_count = head_dim // 2
    base_pairs = pair_count // axis_count
    extra_pairs = pair_count % axis_count
    return tuple(2 * (base_pairs + int(axis_idx < extra_pairs)) for axis_idx in range(axis_count))


def _apply_axis_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE to one even-sized section of query/key heads."""

    pair_count = values.shape[-1] // 2
    omega = torch.arange(pair_count, device=values.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(pair_count - 1, 1)))
    angles = positions.to(device=values.device, dtype=torch.float32)[:, None] * omega[None, :]
    sin = torch.sin(angles)[None, None, :, :]
    cos = torch.cos(angles)[None, None, :, :]

    paired = values.float().reshape(*values.shape[:-1], pair_count, 2)
    x0 = paired[..., 0]
    x1 = paired[..., 1]
    rotated = torch.stack(
        (
            x0 * cos - x1 * sin,
            x0 * sin + x1 * cos,
        ),
        dim=-1,
    ).flatten(-2)
    return rotated.to(dtype=values.dtype)


def _apply_factorized_3d_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    rotary_dims: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply factorized temporal/y/x rotary embeddings to query and key heads."""

    query_parts: list[torch.Tensor] = []
    key_parts: list[torch.Tensor] = []
    offset = 0
    for axis_idx, axis_dim in enumerate(rotary_dims):
        end = offset + axis_dim
        query_parts.append(_apply_axis_rope(query[..., offset:end], positions[:, axis_idx]))
        key_parts.append(_apply_axis_rope(key[..., offset:end], positions[:, axis_idx]))
        offset = end
    if offset < query.shape[-1]:
        query_parts.append(query[..., offset:])
        key_parts.append(key[..., offset:])
    return torch.cat(query_parts, dim=-1), torch.cat(key_parts, dim=-1)


def _factorized_3d_token_positions(
    time_grid: int,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened temporal/y/x token coordinates matching tubelet order."""

    t = torch.arange(time_grid, device=device, dtype=torch.float32)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    grid_t, grid_y, grid_x = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack((grid_t, grid_y, grid_x), dim=-1).reshape(-1, 3)


class RotaryTransformerEncoderLayer(nn.Module):
    """Norm-first transformer encoder layer with factorized 3D RoPE attention."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = "dim must be divisible by num_heads."
            raise ValueError(msg)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.rotary_dims = _split_rotary_dims(self.head_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def _attention(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, token_count, dim = tokens.shape
        if positions.shape != (token_count, 3):
            msg = (
                "positions must have shape "
                f"({token_count}, 3), got {tuple(positions.shape)}."
            )
            raise ValueError(msg)
        qkv = self.qkv(tokens).reshape(
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query, key = _apply_factorized_3d_rope(query, key, positions, self.rotary_dims)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, token_count, dim)
        return self.proj_drop(self.proj(attended))

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Encode tokens using positions only inside query/key attention."""

        tokens = tokens + self._attention(self.norm1(tokens), positions)
        return tokens + self.mlp(self.norm2(tokens))


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


class EventTubeletTransformerEncoder(nn.Module):
    """Tubelet-token transformer for event voxel grids.

    The first `2 * event_bins` channels are interpreted as polarity-by-time
    event bins and embedded with a 3D tubelet convolution. Remaining
    metadata/navigation channels are embedded per spatial patch and added as
    causal auxiliary context.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 192,
        event_bins: int = 5,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.05,
        position_encoding: str = "additive",
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            msg = "embed_dim must be divisible by 4."
            raise ValueError(msg)
        if embed_dim % num_heads != 0:
            msg = "embed_dim must be divisible by num_heads."
            raise ValueError(msg)
        if event_bins <= 0:
            msg = "event_bins must be positive."
            raise ValueError(msg)
        event_channel_count = event_bins * 2
        if in_channels < event_channel_count:
            msg = (
                f"EventTubeletTransformerEncoder expects at least {event_channel_count} "
                f"channels for {event_bins} positive/negative event bins, got {in_channels}."
            )
            raise ValueError(msg)
        if not 1 <= temporal_patch_size <= event_bins:
            msg = "temporal_patch_size must be in [1, event_bins]."
            raise ValueError(msg)
        if position_encoding not in {"additive", "rope"}:
            msg = "position_encoding must be one of {'additive', 'rope'}."
            raise ValueError(msg)
        self.output_dim = embed_dim
        self.patch_size = patch_size
        self.event_bins = event_bins
        self.position_encoding = position_encoding
        self.event_channel_count = event_channel_count
        self.extra_channel_count = in_channels - event_channel_count
        self.event_embed = nn.Conv3d(
            2,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )
        self.aux_embed = (
            nn.Conv2d(
                self.extra_channel_count,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            if self.extra_channel_count
            else None
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [
                (
                    RotaryTransformerEncoderLayer(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    if position_encoding == "rope"
                    else nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=int(embed_dim * mlp_ratio),
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def _tokens_and_positions(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_x = x[:, : self.event_channel_count].view(
            x.shape[0],
            2,
            self.event_bins,
            x.shape[-2],
            x.shape[-1],
        )
        tubelets = self.event_embed(event_x)
        time_grid, grid_h, grid_w = tubelets.shape[-3:]
        tokens = tubelets.permute(0, 2, 3, 4, 1).reshape(
            x.shape[0],
            time_grid * grid_h * grid_w,
            self.output_dim,
        )
        if self.aux_embed is not None:
            aux_tokens = self.aux_embed(x[:, self.event_channel_count :])
            aux_tokens = aux_tokens.flatten(2).transpose(1, 2)
            aux_tokens = aux_tokens[:, None, :, :].expand(
                x.shape[0],
                time_grid,
                aux_tokens.shape[1],
                self.output_dim,
            )
            tokens = tokens + aux_tokens.reshape(
                x.shape[0],
                time_grid * grid_h * grid_w,
                self.output_dim,
            )
        tokens = self.patch_norm(tokens)
        positions = _factorized_3d_token_positions(
            time_grid,
            grid_h,
            grid_w,
            device=tokens.device,
        )
        if self.position_encoding == "additive":
            spatial_pos = _sincos_2d_position_embedding(
                grid_h,
                grid_w,
                self.output_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            temporal_pos = _sincos_1d_position_embedding(
                time_grid,
                self.output_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            pos = (temporal_pos[:, None, :] + spatial_pos[None, :, :]).reshape(
                time_grid * grid_h * grid_w,
                self.output_dim,
            )
            tokens = tokens + pos[None, :, :]
        return tokens, positions

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _positions = self._tokens_and_positions(x)
        return tokens

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into dense spatio-temporal tokens."""

        tokens, positions = self._tokens_and_positions(x)
        for layer in self.layers:
            if isinstance(layer, RotaryTransformerEncoderLayer):
                tokens = layer(tokens, positions)
            else:
                tokens = layer(tokens)
        return self.final_norm(tokens)

    def forward_intermediate_tokens(
        self,
        x: torch.Tensor,
        layer_indices: tuple[int, ...],
    ) -> list[torch.Tensor]:
        """Encode a batch and return selected intermediate token layers."""

        if not layer_indices:
            return [self.forward_tokens(x)]
        depth = len(self.layers)
        if any(layer_idx < 0 or layer_idx >= depth for layer_idx in layer_indices):
            msg = f"layer_indices must be in [0, {depth - 1}], got {layer_indices}."
            raise ValueError(msg)
        selected = set(layer_indices)
        tokens, positions = self._tokens_and_positions(x)
        outputs: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, RotaryTransformerEncoderLayer):
                tokens = layer(tokens, positions)
            else:
                tokens = layer(tokens)
            if layer_idx in selected:
                outputs.append(self.final_norm(tokens))
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into pooled latent vectors."""

        return self.forward_tokens(x).mean(dim=1)


class EventTubeletTransformerRegressor(nn.Module):
    """Tubelet transformer encoder with a log-TTC regression head."""

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 192,
        event_bins: int = 5,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        depth: int = 6,
        num_heads: int = 6,
        position_encoding: str = "additive",
    ) -> None:
        super().__init__()
        self.encoder = EventTubeletTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            event_bins=event_bins,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            depth=depth,
            num_heads=num_heads,
            position_encoding=position_encoding,
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
