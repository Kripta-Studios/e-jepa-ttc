"""Event-only target/background query localization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TargetQueryOutput:
    """Soft localization, query embedding and normalized box moments."""

    object_token: torch.Tensor
    background_token: torch.Tensor
    mask_logits: torch.Tensor
    soft_mask: torch.Tensor
    box_xyxy: torch.Tensor
    object_score: torch.Tensor


def _mask_box(soft_mask: torch.Tensor) -> torch.Tensor:
    batch, height, width = soft_mask.shape
    y = torch.linspace(0.0, 1.0, height, device=soft_mask.device, dtype=soft_mask.dtype)
    x = torch.linspace(0.0, 1.0, width, device=soft_mask.device, dtype=soft_mask.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    denominator = soft_mask.sum(dim=(-1, -2)).clamp_min(1e-6)
    center_x = (soft_mask * xx).sum(dim=(-1, -2)) / denominator
    center_y = (soft_mask * yy).sum(dim=(-1, -2)) / denominator
    variance_x = (soft_mask * (xx - center_x[:, None, None]).square()).sum(
        dim=(-1, -2)
    ) / denominator
    variance_y = (soft_mask * (yy - center_y[:, None, None]).square()).sum(
        dim=(-1, -2)
    ) / denominator
    half_width = (3.0 * variance_x.clamp_min(1e-6)).sqrt().clamp(0.02, 0.5)
    half_height = (3.0 * variance_y.clamp_min(1e-6)).sqrt().clamp(0.02, 0.5)
    return torch.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


class TargetBackgroundQuery(nn.Module):
    """Produce one primary-object and one background token from dense patches."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.target_query = nn.Parameter(torch.empty(dim))
        self.background_query = nn.Parameter(torch.empty(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.normal_(self.target_query, std=0.02)
        nn.init.normal_(self.background_query, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> TargetQueryOutput:
        """Localize from current-frame ``[B,P,D]`` tokens."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,P,D].")
        height, width = spatial_shape
        if tokens.shape[1] != height * width:
            raise ValueError("spatial_shape does not match patch count.")
        keys = self.key(tokens)
        scale = tokens.shape[-1] ** -0.5
        mask_logits = torch.einsum("bpd,d->bp", keys, self.target_query) * scale
        background_logits = torch.einsum("bpd,d->bp", keys, self.background_query) * scale
        soft_mask = torch.sigmoid(mask_logits).reshape(tokens.shape[0], height, width)
        target_weights = torch.softmax(mask_logits, dim=-1)
        background_weights = torch.softmax(background_logits - mask_logits, dim=-1)
        object_token = torch.einsum("bp,bpd->bd", target_weights, tokens)
        background_token = torch.einsum("bp,bpd->bd", background_weights, tokens)
        return TargetQueryOutput(
            object_token=object_token,
            background_token=background_token,
            mask_logits=mask_logits.reshape(tokens.shape[0], height, width),
            soft_mask=soft_mask,
            box_xyxy=_mask_box(soft_mask),
            object_score=self.score(object_token).squeeze(-1),
        )


__all__ = ["TargetBackgroundQuery", "TargetQueryOutput"]
