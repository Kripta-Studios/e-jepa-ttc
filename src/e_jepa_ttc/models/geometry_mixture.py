"""Physical TTC experts and deterministic/router mixtures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.geometry import (
    affine_expansion_inverse_ttc,
    area_rate_inverse_ttc,
    event_contrast_inverse_ttc,
    geometry_track_confidence,
    height_ratio_inverse_ttc,
    weighted_inverse_ttc,
)
from e_jepa_ttc.models.stable_geometry_router import StableGeometryRouter


@dataclass
class GeometryMixtureOutput:
    """Expert predictions and their selected mixture."""

    inverse_ttc: torch.Tensor
    estimates: torch.Tensor
    confidence: torch.Tensor
    weights: torch.Tensor
    router_balance_loss: torch.Tensor
    router_entropy: torch.Tensor


class GeometryMixture(nn.Module):
    """Height, area, affine and event-contrast inverse-TTC experts."""

    def __init__(
        self,
        latent_dim: int,
        *,
        learned_router: bool = True,
        inference_top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.learned_router = learned_router
        self.router = StableGeometryRouter(
            latent_dim + 8,
            4,
            inference_top_k=inference_top_k,
        )

    def forward(
        self,
        *,
        boxes_xyxy: torch.Tensor,
        object_mask: torch.Tensor,
        event_frames: torch.Tensor,
        object_token: torch.Tensor,
        times_s: torch.Tensor,
        soft_masks: torch.Tensor | None = None,
    ) -> GeometryMixtureOutput:
        """Evaluate all experts on causal history only."""

        x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
        widths = (x1 - x0).clamp_min(1e-6)
        heights = (y1 - y0).clamp_min(1e-6)
        areas = widths * heights
        height, height_confidence = height_ratio_inverse_ttc(
            heights,
            times_s,
            valid_mask=object_mask,
        )
        area, area_confidence = area_rate_inverse_ttc(
            areas,
            times_s,
            valid_mask=object_mask,
        )
        affine, affine_confidence = affine_expansion_inverse_ttc(
            boxes_xyxy,
            times_s,
            valid_mask=object_mask,
        )
        contrast, contrast_confidence = event_contrast_inverse_ttc(
            event_frames,
            times_s,
            soft_masks=soft_masks,
        )
        if contrast.ndim == 1 and height.ndim == 2:
            contrast = contrast[:, None].expand_as(height)
            contrast_confidence = contrast_confidence[:, None].expand_as(height)
        track_confidence = geometry_track_confidence(boxes_xyxy, object_mask)
        confidence = torch.stack(
            (
                height_confidence,
                area_confidence,
                affine_confidence,
                contrast_confidence,
            ),
            dim=-1,
        )
        confidence = confidence * track_confidence[..., None]
        estimates = torch.stack((height, area, affine, contrast), dim=-1)
        if self.learned_router:
            current_width = widths[:, -1]
            current_height = heights[:, -1]
            quality = torch.cat(
                (
                    current_width[..., None],
                    current_height[..., None],
                    confidence,
                    estimates.clamp(0.0, 4.0).mean(dim=-1, keepdim=True),
                    track_confidence[..., None],
                ),
                dim=-1,
            )
            if object_token.ndim == 2:
                object_token = object_token[:, None]
            routing = self.router(
                torch.cat((object_token, quality), dim=-1),
                estimates,
                confidence,
            )
            return GeometryMixtureOutput(
                inverse_ttc=routing.inverse_ttc,
                estimates=estimates,
                confidence=confidence,
                weights=routing.weights,
                router_balance_loss=routing.balance_loss,
                router_entropy=routing.entropy,
            )
        inverse_ttc, _ = weighted_inverse_ttc(estimates, confidence)
        weights = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        zero = inverse_ttc.new_zeros(())
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return GeometryMixtureOutput(
            inverse_ttc=inverse_ttc,
            estimates=estimates,
            confidence=confidence,
            weights=weights,
            router_balance_loss=zero,
            router_entropy=entropy,
        )


__all__ = ["GeometryMixture", "GeometryMixtureOutput"]
