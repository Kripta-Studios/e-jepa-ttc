"""Object Event TTC v4.7: high-resolution mask extent geometry.

V4.6 showed that foreground can generalise while a content-dependent scale head
cannot. V4.7 therefore predicts a full-resolution foreground mask and derives the
height ratio directly from its differentiable vertical extent. No boxes, heights,
sequence IDs, or motion features are accepted by ``forward``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


@dataclass(frozen=True)
class ObjectEventV47Config:
    decoder_hidden_dim: int = 96
    mask_size: int = 64
    foreground_temperature: float = 1.0
    edge_temperature: float = 0.08
    moment_floor: float = 1.0e-4
    event_skip_epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        if min(self.decoder_hidden_dim, self.mask_size) <= 0:
            raise ValueError("v4.7 dimensions must be positive")
        if self.mask_size < 16:
            raise ValueError("v4.7 mask_size must be at least 16")
        if min(
            self.foreground_temperature,
            self.edge_temperature,
            self.moment_floor,
            self.event_skip_epsilon,
        ) <= 0.0:
            raise ValueError("v4.7 numerical scales must be positive")


@dataclass
class ObjectEventV47Output:
    expansion: torch.Tensor
    reverse_expansion: torch.Tensor
    raw_score: torch.Tensor
    reverse_raw_score: torch.Tensor
    reversal_consistency_error: torch.Tensor
    foreground_logits: torch.Tensor
    foreground_probabilities: torch.Tensor
    predicted_log_heights: torch.Tensor
    height_log_eta: torch.Tensor
    row_profiles: torch.Tensor
    top_positions: torch.Tensor
    bottom_positions: torch.Tensor
    event_activity: torch.Tensor


class ObjectEventTTCV47(nn.Module):
    """High-resolution event-only foreground extent model."""

    def __init__(
        self,
        base_config: ObjectEventV41Config | None = None,
        geometry_config: ObjectEventV47Config | None = None,
    ) -> None:
        super().__init__()
        self.config = base_config or ObjectEventV41Config()
        self.geometry_config = geometry_config or ObjectEventV47Config()
        template = ObjectEventTTCV42(self.config)
        self.geometry_encoder = copy.deepcopy(template.encoder)
        hidden = self.geometry_config.decoder_hidden_dim
        channels = self.config.embed_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden // 2), hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )

    def load_v46_geometry_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        prefix = "geometry_encoder."
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not extracted:
            raise ValueError("v4.6 checkpoint contains no geometry_encoder parameters")
        self.geometry_encoder.load_state_dict(extracted, strict=True)

    def _resize(self, events: torch.Tensor) -> torch.Tensor:
        expected = (self.config.temporal_steps, self.config.in_channels)
        if events.ndim != 5 or events.shape[1:3] != expected:
            raise ValueError(
                f"events must be [B,{expected[0]},{expected[1]},H,W], got {tuple(events.shape)}"
            )
        if events.shape[-2:] == (self.config.input_size, self.config.input_size):
            return events
        batch, steps, channels, height, width = events.shape
        resized = functional.interpolate(
            events.reshape(batch * steps, channels, height, width).float(),
            size=(self.config.input_size, self.config.input_size),
            mode="area",
        )
        return resized.reshape(
            batch,
            steps,
            channels,
            self.config.input_size,
            self.config.input_size,
        ).to(events.dtype)

    def _normalised_activity(self, resized: torch.Tensor) -> torch.Tensor:
        activity = torch.log1p(resized.float().abs()).mean(dim=2)
        mean = activity.mean(dim=(-2, -1), keepdim=True)
        std = activity.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (activity - mean) / std.clamp_min(self.geometry_config.event_skip_epsilon)

    @staticmethod
    def soft_vertical_extent(
        probabilities: torch.Tensor,
        *,
        edge_temperature: float,
        floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return differentiable row profile, top, bottom and vertical extent.

        The row profile is the mean foreground probability over width. Soft top
        and bottom operators focus on occupied rows while preserving sub-pixel
        gradients. A global multiplicative scale cancels in the temporal ratio.
        """

        if probabilities.ndim != 4 or probabilities.shape[1] != 1:
            raise ValueError("probabilities must be [N,1,H,W]")
        rows = probabilities[:, 0].float().mean(dim=-1).clamp_min(floor)
        height = rows.shape[-1]
        y = torch.linspace(0.0, 1.0, height, device=rows.device, dtype=rows.dtype)
        log_mass = torch.log(rows)
        top_weights = torch.softmax(log_mass - y[None] / edge_temperature, dim=-1)
        bottom_weights = torch.softmax(log_mass + y[None] / edge_temperature, dim=-1)
        top = (top_weights * y[None]).sum(dim=-1)
        bottom = (bottom_weights * y[None]).sum(dim=-1)
        extent = (bottom - top).clamp_min(floor)
        return rows, top, bottom, extent

    def _forward_geometry(self, events: torch.Tensor) -> tuple[torch.Tensor, ...]:
        resized = self._resize(events)
        batch, steps, channels, height, width = resized.shape
        maps = self.geometry_encoder(
            resized.reshape(batch * steps, channels, height, width)
        )
        decoded = self.decoder(maps)
        decoded = functional.interpolate(
            decoded,
            size=(self.geometry_config.mask_size, self.geometry_config.mask_size),
            mode="bilinear",
            align_corners=False,
        )
        activity = self._normalised_activity(resized)
        activity = functional.interpolate(
            activity.reshape(batch * steps, 1, height, width),
            size=(self.geometry_config.mask_size, self.geometry_config.mask_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.refine(torch.cat((decoded, activity), dim=1))
        probabilities = torch.sigmoid(
            logits / self.geometry_config.foreground_temperature
        )
        rows, top, bottom, extent = self.soft_vertical_extent(
            probabilities,
            edge_temperature=self.geometry_config.edge_temperature,
            floor=self.geometry_config.moment_floor,
        )
        log_heights = torch.log(extent).reshape(batch, steps)
        logits = logits.reshape(batch, steps, self.geometry_config.mask_size, self.geometry_config.mask_size)
        probabilities = probabilities.reshape_as(logits)
        rows = rows.reshape(batch, steps, self.geometry_config.mask_size)
        top = top.reshape(batch, steps)
        bottom = bottom.reshape(batch, steps)
        activity = activity.reshape(batch, steps, self.geometry_config.mask_size, self.geometry_config.mask_size)
        return log_heights, logits, probabilities, rows, top, bottom, activity

    def forward(self, events: torch.Tensor) -> ObjectEventV47Output:
        (
            log_heights,
            logits,
            probabilities,
            row_profiles,
            top,
            bottom,
            activity,
        ) = self._forward_geometry(events)
        height_log_eta = log_heights[:, 1] - log_heights[:, 2]
        maximum = self.config.max_abs_expansion
        expansion = (1.0 - torch.exp(height_log_eta)).clamp(
            -maximum * 0.999, maximum * 0.999
        )
        reverse_log_eta = -height_log_eta
        reverse_expansion = (1.0 - torch.exp(reverse_log_eta)).clamp(
            -maximum * 0.999, maximum * 0.999
        )
        raw_score = torch.atanh((expansion / maximum).clamp(-0.999, 0.999))
        reverse_raw_score = torch.atanh(
            (reverse_expansion / maximum).clamp(-0.999, 0.999)
        )
        return ObjectEventV47Output(
            expansion=expansion,
            reverse_expansion=reverse_expansion,
            raw_score=raw_score,
            reverse_raw_score=reverse_raw_score,
            reversal_consistency_error=(height_log_eta + reverse_log_eta).abs(),
            foreground_logits=logits,
            foreground_probabilities=probabilities,
            predicted_log_heights=log_heights,
            height_log_eta=height_log_eta,
            row_profiles=row_profiles,
            top_positions=top,
            bottom_positions=bottom,
            event_activity=activity,
        )


__all__ = ["ObjectEventTTCV47", "ObjectEventV47Config", "ObjectEventV47Output"]
