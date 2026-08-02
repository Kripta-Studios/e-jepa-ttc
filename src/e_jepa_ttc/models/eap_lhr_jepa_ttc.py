"""Leakage-safe full-frame + object-ROI LHR object-JEPA TTC model."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models import build_encoder
from e_jepa_ttc.models.dense_patch_ttc import DensePatchEventBackbone


@dataclass(frozen=True)
class EAPLHRJEPATTCConfig:
    endpoint_event_channels: int = 20
    observable_motion_dim: int = 18
    geometry_target_dim: int = 20
    category_count: int = 4
    dim: int = 256
    use_rgb: bool = False
    ttc_residual_scale_s: float = 0.25
    foreground_size: int = 128
    use_observable_motion: bool = True


@dataclass
class EAPLHRJEPATTCOutput:
    ttc_seconds: torch.Tensor
    lhr_ttc_seconds: torch.Tensor
    predicted_heights: torch.Tensor
    predicted_height_ratio: torch.Tensor
    geometry_prediction: torch.Tensor
    category_logits: torch.Tensor
    foreground_logits: torch.Tensor
    jepa_prediction: torch.Tensor
    jepa_target: torch.Tensor
    object_token: torch.Tensor


class _EndpointEncoder(nn.Module):
    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        self.backbone = DensePatchEventBackbone(channels, dim=dim)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        encoded = self.backbone(values[:, None])
        return encoded.global_token[:, 0], encoded.dense_tokens[:, 0], encoded.spatial_shape


class _RGBPairEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 5 or tuple(pair.shape[1:3]) != (2, 3):
            raise ValueError("RGB pair must have shape [B,2,3,H,W].")
        values = pair.float()
        if float(values.detach().max()) > 1.5:
            values = values / 255.0
        return self.network(values.flatten(1, 2)).flatten(1)


class EAPLHRJEPATTC(nn.Module):
    """Complete TTC estimator retained unchanged for zero-shot transfer.

    Only observable causal inputs enter ``forward``:
    event full frames, event ROIs, box-derived 2-D motion, time gap, and
    optionally RGB. TTC, depth, 3-D geometry and category are supervision only.
    """

    def __init__(self, config: EAPLHRJEPATTCConfig) -> None:
        super().__init__()
        self.config = config
        self.full_encoder = build_encoder("event-tubelet-transformer", in_channels=21)
        full_dim = int(self.full_encoder.output_dim)
        self.full_projection = nn.Linear(full_dim, config.dim)
        self.roi_encoder = _EndpointEncoder(config.endpoint_event_channels, config.dim)
        self.target_roi_encoder = copy.deepcopy(self.roi_encoder).requires_grad_(False)
        self.target_roi_encoder.eval()

        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(config.observable_motion_dim),
            nn.Linear(config.observable_motion_dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(1, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        self.rgb_encoder = _RGBPairEncoder(config.dim) if config.use_rgb else None

        # Strictly causal JEPA: ROI(t1) + full-frame(t1) + motion(t0->t1) + dt.
        self.jepa_predictor = nn.Sequential(
            nn.LayerNorm(config.dim * 4),
            nn.Linear(config.dim * 4, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
        )

        fusion_count = 6 + int(config.use_rgb)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.dim * fusion_count),
            nn.Linear(config.dim * fusion_count, config.dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.dim * 2, config.dim),
        )
        self.height_head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, 2),
        )
        self.ttc_residual_head = nn.Linear(config.dim, 1)
        self.geometry_head = nn.Linear(config.dim, config.geometry_target_dim)
        self.category_head = nn.Linear(config.dim, config.category_count)
        self.foreground_head = nn.Linear(config.dim, 2)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must lie in [0,1].")
        for target, online in zip(
            self.target_roi_encoder.parameters(),
            self.roi_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)

    def load_geo_encoder(self, checkpoint: dict[str, object]) -> None:
        state = checkpoint.get("encoder_state_dict")
        if not isinstance(state, dict):
            raise ValueError("eAP-Geo checkpoint has no encoder_state_dict.")
        self.full_encoder.load_state_dict(state, strict=True)

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith(("target_roi_encoder.", "jepa_predictor."))
        }

    def _full_tokens(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim == 4:
            tokens = self.full_projection(self.full_encoder.forward_tokens(values).mean(1))
            return tokens, tokens, torch.zeros_like(tokens)
        if values.ndim != 5 or values.shape[1] < 2:
            raise ValueError("full_frame_events must be [B,21,H,W] or [B,T,21,H,W].")
        first = values[:, -2]
        second = values[:, -1]
        joined = torch.cat((first, second), dim=0)
        encoded = self.full_projection(self.full_encoder.forward_tokens(joined).mean(1))
        first_token, second_token = encoded.chunk(2, dim=0)
        return first_token, second_token, second_token - first_token

    def forward(
        self,
        *,
        full_frame_events: torch.Tensor,
        event_roi_pair: torch.Tensor,
        delta_t_s: torch.Tensor,
        observable_motion: torch.Tensor,
        jepa_context_motion: torch.Tensor | None = None,
        rgb_pair: torch.Tensor | None = None,
    ) -> EAPLHRJEPATTCOutput:
        if event_roi_pair.ndim != 4:
            raise ValueError("event_roi_pair must have shape [B,2*C,H,W].")
        expected = self.config.endpoint_event_channels * 2
        if event_roi_pair.shape[1] != expected:
            raise ValueError(f"Expected {expected} ROI channels, got {event_roi_pair.shape[1]}.")
        if (
            observable_motion.ndim != 2
            or observable_motion.shape[1] != self.config.observable_motion_dim
        ):
            raise ValueError(
                f"observable_motion must have shape [B,{self.config.observable_motion_dim}]."
            )

        first_full_token, full_token, full_delta = self._full_tokens(full_frame_events)
        first_values, second_values = event_roi_pair.chunk(2, dim=1)
        first_token, _first_dense, _shape = self.roi_encoder(first_values)
        second_token, second_dense, spatial_shape = self.roi_encoder(second_values)
        with torch.no_grad():
            target_token, _target_dense, _target_shape = self.target_roi_encoder(second_values)

        effective_motion = (
            observable_motion
            if self.config.use_observable_motion
            else torch.zeros_like(observable_motion)
        )
        motion_token = self.motion_encoder(effective_motion)
        if jepa_context_motion is None:
            jepa_context_motion = torch.zeros_like(observable_motion)
        jepa_motion_token = self.motion_encoder(jepa_context_motion)
        elapsed = delta_t_s.reshape(-1, 1).clamp_min(1e-6)
        delta_token = self.delta_encoder(torch.log1p(elapsed))

        jepa_condition = torch.cat(
            (first_token, first_full_token, jepa_motion_token, delta_token),
            dim=-1,
        )
        jepa_prediction = self.jepa_predictor(jepa_condition)

        pieces = [
            full_token,
            full_delta,
            first_token,
            second_token,
            second_token - first_token,
            motion_token,
        ]
        if self.rgb_encoder is not None:
            if rgb_pair is None:
                raise ValueError("Configured RGB model requires rgb_pair.")
            pieces.append(self.rgb_encoder(rgb_pair))
        object_token = self.fusion(torch.cat(pieces, dim=-1))

        predicted_heights = functional.softplus(self.height_head(object_token)) + 1e-3
        ratio = predicted_heights[:, 0] / predicted_heights[:, 1].clamp_min(1e-3)
        denominator = 1.0 - ratio
        safe_denominator = torch.where(
            denominator.abs() < 1e-3,
            denominator.sign().masked_fill(denominator == 0, 1.0) * 1e-3,
            denominator,
        )
        delta_flat = elapsed[:, 0]
        lhr_ttc = delta_flat / safe_denominator
        residual = self.config.ttc_residual_scale_s * torch.tanh(
            self.ttc_residual_head(object_token).squeeze(-1)
        )
        ttc = lhr_ttc + residual

        patch_logits = self.foreground_head(second_dense)
        height, width = spatial_shape
        foreground = patch_logits.transpose(1, 2).reshape(-1, 2, height, width)
        foreground = functional.interpolate(
            foreground,
            size=(self.config.foreground_size, self.config.foreground_size),
            mode="bilinear",
            align_corners=False,
        )
        return EAPLHRJEPATTCOutput(
            ttc_seconds=ttc,
            lhr_ttc_seconds=lhr_ttc,
            predicted_heights=predicted_heights,
            predicted_height_ratio=ratio,
            geometry_prediction=self.geometry_head(object_token),
            category_logits=self.category_head(object_token),
            foreground_logits=foreground,
            jepa_prediction=jepa_prediction,
            jepa_target=target_token.detach(),
            object_token=object_token,
        )
