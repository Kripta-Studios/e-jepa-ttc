"""RGB fusion and frozen-foundation-teacher distillation for Event-JEPA TTC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    ObjectJEPAConfig,
    normalized_box_features,
)


@dataclass
class MultimodalTTCOutput:
    """Fused distribution and modality-specific representations."""

    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    risk_logits: torch.Tensor
    object_mask: torch.Tensor
    event_tokens: torch.Tensor
    rgb_tokens: torch.Tensor
    fusion_gate: torch.Tensor


class RGBRecurrentObjectEncoder(nn.Module):
    """Compact causal RGB encoder for pre-cropped tracked-object histories."""

    def __init__(self, config: ObjectJEPAConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        width = config.feature_dim
        self.config = config
        self.spatial = nn.Sequential(
            nn.Conv2d(3, width // 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.visual_projection = nn.Sequential(nn.Linear(width, dim), nn.GELU())
        self.geometry_projection = nn.Sequential(nn.Linear(6, dim), nn.GELU())
        self.recurrent_cell = nn.GRUCell(dim * 2, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
        )

    def forward(
        self,
        context_rgb: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one recurrent RGB token per pre-cropped object sample."""

        if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
            msg = "context_rgb must have shape [B,T,3,H,W]."
            raise ValueError(msg)
        if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[2] != 1:
            msg = "Pre-cropped RGB currently requires boxes [B,T,1,4]."
            raise ValueError(msg)
        if object_mask.shape != boxes_xyxy.shape[:3]:
            msg = "RGB box and object mask shapes must match."
            raise ValueError(msg)
        batch, steps = context_rgb.shape[:2]
        if boxes_xyxy.shape[:2] != (batch, steps):
            msg = "RGB and box temporal shapes must match."
            raise ValueError(msg)
        hidden = context_rgb.new_zeros((batch, self.config.embedding_dim), dtype=torch.float32)
        for step in range(steps):
            image = context_rgb[:, step].to(dtype=torch.float32)
            if context_rgb.dtype == torch.uint8:
                image = image / 255.0
            image = (image - self.image_mean) / self.image_std
            visual = self.spatial(image).mean(dim=(-1, -2))
            visual = self.visual_projection(visual)
            geometry = self.geometry_projection(
                normalized_box_features(boxes_xyxy[:, step, 0].to(dtype=torch.float32))
            )
            update = self.recurrent_cell(torch.cat((visual, geometry), dim=-1), hidden)
            hidden = torch.where(object_mask[:, step, 0, None].bool(), update, hidden)
        return self.output_norm(hidden)[:, None, :] * object_mask[:, -1, :, None]


class ObjectEventRGBFusion(nn.Module):
    """Gated late fusion that preserves an event-only deployable branch."""

    def __init__(self, event_model: ObjectCentricEventJEPA) -> None:
        super().__init__()
        self.event_model = event_model
        self.config = event_model.config
        self.rgb_encoder = RGBRecurrentObjectEncoder(self.config)
        dim = self.config.embedding_dim
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(dim)
        self.inverse_ttc_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.risk_head = nn.Linear(dim, len(self.config.risk_thresholds_s))

    def forward(
        self,
        context_events: torch.Tensor,
        context_rgb: torch.Tensor,
        context_boxes: torch.Tensor,
        context_object_mask: torch.Tensor,
        *,
        context_sampling_boxes: torch.Tensor | None = None,
        context_ego_actions: torch.Tensor | None = None,
        context_ego_action_mask: torch.Tensor | None = None,
    ) -> MultimodalTTCOutput:
        """Fuse causal recurrent RGB and event tokens for TTC prediction."""

        event = self.event_model.context_encoder(
            context_events,
            context_boxes,
            context_object_mask,
            sampling_boxes_xyxy=context_sampling_boxes,
            ego_actions=context_ego_actions,
            ego_action_mask=context_ego_action_mask,
        )
        rgb_tokens = self.rgb_encoder(context_rgb, context_boxes, context_object_mask)
        gate = self.fusion_gate(torch.cat((event.object_tokens, rgb_tokens), dim=-1))
        fused = self.fusion_norm(gate * event.object_tokens + (1.0 - gate) * rgb_tokens)
        distribution = self.inverse_ttc_head(fused)
        return MultimodalTTCOutput(
            inverse_ttc_mean=distribution[..., 0],
            inverse_ttc_log_variance=distribution[..., 1].clamp(-8.0, 5.0),
            risk_logits=self.risk_head(fused),
            object_mask=event.object_mask,
            event_tokens=event.object_tokens,
            rgb_tokens=rgb_tokens,
            fusion_gate=gate,
        )


def multimodal_ttc_loss(
    output: MultimodalTTCOutput,
    ttc_target_s: torch.Tensor,
    *,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    risk_weight: float = 0.25,
    rgb_to_event_distillation_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Train fused TTC and distill detached RGB evidence into the event branch."""

    valid = output.object_mask & torch.isfinite(ttc_target_s) & (ttc_target_s.abs() >= 0.1)
    if not torch.any(valid):
        msg = "Multimodal TTC loss requires at least one valid target."
        raise ValueError(msg)
    target = torch.reciprocal(ttc_target_s[valid])
    residual = output.inverse_ttc_mean[valid] - target
    log_variance = output.inverse_ttc_log_variance[valid]
    nll = (0.5 * torch.exp(-log_variance) * residual.square() + 0.5 * log_variance).mean()
    thresholds = output.risk_logits.new_tensor(risk_thresholds_s)
    labels = (
        (ttc_target_s[..., None] > 0)
        & (ttc_target_s[..., None] <= thresholds[None, None, :])
    ).to(output.risk_logits.dtype)
    risk = functional.binary_cross_entropy_with_logits(output.risk_logits[valid], labels[valid])
    student = functional.normalize(output.event_tokens[valid], dim=-1)
    teacher = functional.normalize(output.rgb_tokens[valid].detach(), dim=-1)
    distillation = (1.0 - (student * teacher).sum(dim=-1)).mean()
    total = nll + risk_weight * risk + rgb_to_event_distillation_weight * distillation
    return {
        "total": total,
        "inverse_ttc_nll": nll,
        "risk_bce": risk,
        "rgb_to_event_distillation": distillation,
    }


class DINOv3FeatureTeacher(nn.Module):
    """Lazy frozen DINOv3 adapter for offline RGB feature distillation."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ) -> None:
        super().__init__()
        try:
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel
        except ImportError as error:
            msg = "Install the optional 'multimodal' dependency group for DINOv3 distillation."
            raise RuntimeError(msg) from error
        processor_path = hf_hub_download(model_name, "preprocessor_config.json")
        processor = json.loads(Path(processor_path).read_text(encoding="utf-8"))
        size = processor.get("size", {"height": 224, "width": 224})
        self.input_size = (
            int(size.get("height", size.get("shortest_edge", 224))),
            int(size.get("width", size.get("shortest_edge", 224))),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(processor.get("image_mean", (0.485, 0.456, 0.406))).reshape(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(processor.get("image_std", (0.229, 0.224, 0.225))).reshape(
                1, 3, 1, 1
            ),
        )
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> DINOv3FeatureTeacher:
        """Keep the foundation teacher frozen and in evaluation mode."""

        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """Return pooled per-frame DINOv3 features for ``[B,T,3,H,W]`` RGB."""

        if rgb_uint8.ndim != 5 or rgb_uint8.shape[2] != 3:
            msg = "DINOv3 RGB input must have shape [B,T,3,H,W]."
            raise ValueError(msg)
        batch, steps = rgb_uint8.shape[:2]
        images = rgb_uint8.reshape(batch * steps, *rgb_uint8.shape[2:]).float() / 255.0
        images = functional.interpolate(
            images,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        output = self.backbone(pixel_values=(images - self.image_mean) / self.image_std)
        features = output.last_hidden_state
        if features.ndim == 4:
            features = features.mean(dim=(-1, -2))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        else:
            msg = "Unsupported DINOv3 feature tensor rank."
            raise RuntimeError(msg)
        return features.reshape(batch, steps, -1)


__all__ = [
    "DINOv3FeatureTeacher",
    "MultimodalTTCOutput",
    "ObjectEventRGBFusion",
    "RGBRecurrentObjectEncoder",
    "multimodal_ttc_loss",
]
