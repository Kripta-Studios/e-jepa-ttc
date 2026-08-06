"""Object Event TTC v4.6: learned foreground height-ratio residual.

The v4.5 loss-only experiment improved correlation but did not materially improve
paper-aligned MiD.  V4.6 changes the representation while preserving an auditable
contract: inference receives only the common-coordinate event tensor.  Official
boxes and visible heights are used only as training targets.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42


def _small_linear(module: nn.Linear, *, std: float = 1.0e-3, bias: float = 0.0) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=std)
    nn.init.constant_(module.bias, bias)


@dataclass(frozen=True)
class ObjectEventV46Config:
    foreground_hidden_dim: int = 96
    scale_hidden_dim: int = 128
    maximum_blend: float = 0.75
    residual_log_eta_limit: float = 0.20
    foreground_temperature: float = 1.0
    initial_blend_logit: float = -4.0
    moment_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        if min(self.foreground_hidden_dim, self.scale_hidden_dim) <= 0:
            raise ValueError("v4.6 hidden dimensions must be positive")
        if not 0.0 < self.maximum_blend <= 1.0:
            raise ValueError("maximum_blend must lie in (0,1]")
        if not 0.0 < self.residual_log_eta_limit < 1.0:
            raise ValueError("residual_log_eta_limit must lie in (0,1)")
        if self.foreground_temperature <= 0.0 or self.moment_floor <= 0.0:
            raise ValueError("v4.6 scales must be positive")


@dataclass
class ObjectEventV46Output:
    expansion: torch.Tensor
    reverse_expansion: torch.Tensor
    raw_score: torch.Tensor
    reverse_raw_score: torch.Tensor
    reversal_consistency_error: torch.Tensor
    endpoint_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor
    base_expansion: torch.Tensor
    height_expansion: torch.Tensor
    base_log_eta: torch.Tensor
    height_log_eta: torch.Tensor
    fused_log_eta: torch.Tensor
    foreground_logits: torch.Tensor
    foreground_probabilities: torch.Tensor
    predicted_log_heights: torch.Tensor
    blend: torch.Tensor
    geometry_confidence: torch.Tensor


class ObjectEventTTCV46(nn.Module):
    """Frozen v4.2 baseline plus a trainable event-only height-ratio branch.

    The baseline and geometry encoders are separate copies.  This prevents a new
    geometry objective from silently corrupting the validated v4.2 predictor.
    """

    def __init__(
        self,
        base_config: ObjectEventV41Config | None = None,
        geometry_config: ObjectEventV46Config | None = None,
    ) -> None:
        super().__init__()
        self.config = base_config or ObjectEventV41Config()
        self.geometry_config = geometry_config or ObjectEventV46Config()
        self.base = ObjectEventTTCV42(self.config)
        self.force_base_only = False
        self.geometry_encoder = copy.deepcopy(self.base.encoder)

        channels = self.config.embed_dim
        hidden = self.geometry_config.foreground_hidden_dim
        self.foreground_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        pooled_dim = channels + 4
        self.scale_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.geometry_config.scale_hidden_dim),
            nn.GELU(),
            nn.Linear(self.geometry_config.scale_hidden_dim, 1),
        )
        self.blend_head = nn.Sequential(
            nn.LayerNorm(2 * channels + 5),
            nn.Linear(2 * channels + 5, self.geometry_config.scale_hidden_dim),
            nn.GELU(),
            nn.Linear(self.geometry_config.scale_hidden_dim, 1),
        )
        _small_linear(self.scale_head[-1])
        _small_linear(
            self.blend_head[-1],
            std=1.0e-4,
            bias=self.geometry_config.initial_blend_logit,
        )

    def set_base_only(self, enabled: bool) -> None:
        self.force_base_only = bool(enabled)

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict, strict=True)
        self.geometry_encoder.load_state_dict(self.base.encoder.state_dict(), strict=True)
        self.freeze_base()

    @staticmethod
    def _soft_extent(
        probabilities: torch.Tensor,
        *,
        floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if probabilities.ndim != 4 or probabilities.shape[1] != 1:
            raise ValueError("foreground probabilities must be [N,1,H,W]")
        n, _, height, width = probabilities.shape
        weights = probabilities[:, 0].float().clamp_min(floor)
        mass = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(floor)
        weights = weights / mass
        y = torch.linspace(-1.0, 1.0, height, device=weights.device, dtype=weights.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=weights.device, dtype=weights.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_y = (weights * yy).sum(dim=(-2, -1))
        center_x = (weights * xx).sum(dim=(-2, -1))
        var_y = (weights * (yy[None] - center_y[:, None, None]).square()).sum(dim=(-2, -1))
        var_x = (weights * (xx[None] - center_x[:, None, None]).square()).sum(dim=(-2, -1))
        # 4*std approximates a full foreground extent while remaining smooth.
        height_extent = 4.0 * torch.sqrt(var_y.clamp_min(floor))
        width_extent = 4.0 * torch.sqrt(var_x.clamp_min(floor))
        return height_extent, width_extent, center_y, center_x

    def _geometry_forward(
        self, events: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        resized = self.base._resize(events)
        batch, steps, channels, height, width = resized.shape
        maps = self.geometry_encoder(
            resized.reshape(batch * steps, channels, height, width)
        )
        _, embed_dim, map_h, map_w = maps.shape
        maps = maps.reshape(batch, steps, embed_dim, map_h, map_w)
        flat_maps = maps.reshape(batch * steps, embed_dim, map_h, map_w)
        logits = self.foreground_head(flat_maps)
        probabilities = torch.sigmoid(
            logits / self.geometry_config.foreground_temperature
        )
        h_extent, w_extent, center_y, center_x = self._soft_extent(
            probabilities, floor=self.geometry_config.moment_floor
        )
        weights = probabilities[:, 0].float().clamp_min(self.geometry_config.moment_floor)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(
            self.geometry_config.moment_floor
        )
        pooled = (flat_maps.float() * weights[:, None]).sum(dim=(-2, -1))
        scalar_features = torch.stack((h_extent, w_extent, center_y, center_x), dim=-1)
        correction = self.scale_head(torch.cat((pooled, scalar_features), dim=-1)).squeeze(-1)
        predicted_log_heights = (
            torch.log(h_extent.clamp_min(self.geometry_config.moment_floor)) + correction
        ).reshape(batch, steps)
        logits = logits.reshape(batch, steps, map_h, map_w)
        probabilities = probabilities.reshape(batch, steps, map_h, map_w)
        pooled = pooled.reshape(batch, steps, embed_dim)
        scalar_features = scalar_features.reshape(batch, steps, 4)
        # Official supervision provides visible heights for t1 and t2.
        height_log_eta = predicted_log_heights[:, 1] - predicted_log_heights[:, 2]
        confidence = probabilities[:, 1:3].mean(dim=(-2, -1)).mean(dim=1)
        temporal_feature = torch.cat(
            (
                pooled[:, 1],
                pooled[:, 2] - pooled[:, 1],
                scalar_features[:, 2] - scalar_features[:, 1],
                confidence[:, None],
            ),
            dim=-1,
        )
        blend = self.geometry_config.maximum_blend * torch.sigmoid(
            self.blend_head(temporal_feature).squeeze(-1)
        )
        if self.force_base_only:
            blend = torch.zeros_like(blend)
        return (
            height_log_eta,
            logits,
            probabilities,
            predicted_log_heights,
            blend,
            confidence,
        )

    def forward(self, events: torch.Tensor) -> ObjectEventV46Output:
        with torch.no_grad():
            base_output = self.base(events)
        (
            height_log_eta,
            logits,
            probabilities,
            predicted_log_heights,
            blend,
            confidence,
        ) = self._geometry_forward(events)
        maximum = self.config.max_abs_expansion
        base_expansion = base_output.expansion.detach()
        base_log_eta = torch.log1p(
            -base_expansion.clamp(-maximum * 0.999, maximum * 0.999)
        )
        residual = (height_log_eta - base_log_eta).clamp(
            -self.geometry_config.residual_log_eta_limit,
            self.geometry_config.residual_log_eta_limit,
        )
        fused_log_eta = base_log_eta + blend * residual
        expansion = (1.0 - torch.exp(fused_log_eta)).clamp(
            -maximum * 0.999, maximum * 0.999
        )
        height_expansion = (1.0 - torch.exp(height_log_eta)).clamp(
            -maximum * 0.999, maximum * 0.999
        )
        # The geometry ratio is reciprocal by construction.  The frozen base
        # reverse prediction is kept for compatibility and auditing.
        reverse_expansion = base_output.reverse_expansion.detach()
        raw_score = -fused_log_eta
        reverse_raw_score = base_output.reverse_raw_score.detach()
        return ObjectEventV46Output(
            expansion=expansion,
            reverse_expansion=reverse_expansion,
            raw_score=raw_score,
            reverse_raw_score=reverse_raw_score,
            reversal_consistency_error=(
                torch.log1p(-expansion) + torch.log1p(-reverse_expansion)
            ).abs(),
            endpoint_embeddings=base_output.endpoint_embeddings.detach(),
            spatial_embeddings=base_output.spatial_embeddings.detach(),
            base_expansion=base_expansion,
            height_expansion=height_expansion,
            base_log_eta=base_log_eta,
            height_log_eta=height_log_eta,
            fused_log_eta=fused_log_eta,
            foreground_logits=logits,
            foreground_probabilities=probabilities,
            predicted_log_heights=predicted_log_heights,
            blend=blend,
            geometry_confidence=confidence,
        )


__all__ = [
    "ObjectEventTTCV46",
    "ObjectEventV46Config",
    "ObjectEventV46Output",
]
