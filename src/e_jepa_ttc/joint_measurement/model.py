"""V18 reference kernels and a compact P2 predictor; not a production trainer.

The encoder is injected. Production integration must use a private copy of the
admitted native CNN and the native phase/TTC converter. No dataset, parent path,
optimizer, checkpoint, or unverified model download is hidden in this module.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC


def local_cosine_volume(
    previous: Tensor, current: Tensor, radius: int = 2
) -> tuple[Tensor, Tensor]:
    """Cosine per displacement, with explicit support. Order: row-major dy,dx.

    volume[b,k,y,x] compares previous[y,x] with current[y+dy,x+dx].
    Invalid offsets are zeroed AND accompanied by a mask; zero is not evidence.
    This is a reference implementation, not the optimized production kernel.
    """
    if previous.ndim != 4 or previous.shape != current.shape:
        raise ValueError("Paired maps must have equal [B,C,H,W] shapes")
    if not previous.is_floating_point() or not current.is_floating_point():
        raise TypeError("Floating point feature maps required")
    if previous.device != current.device or radius < 0 or radius > 4:
        raise ValueError("Device mismatch or unregistered radius")
    batch, channels, height, width = previous.shape
    if min(batch, channels, height, width) < 1:
        raise ValueError("Empty feature map")
    # Correlation arithmetic is explicitly FP32 even when an outer autocast exists.
    with torch.autocast(device_type=previous.device.type, enabled=False):
        a = functional.normalize(previous.float(), dim=1, eps=1e-6)
        b = functional.normalize(current.float(), dim=1, eps=1e-6)
        kernel = 2 * radius + 1
        patches = functional.unfold(b, kernel_size=kernel, padding=radius)
        patches = patches.reshape(batch, channels, kernel * kernel, height, width)
        volume = (a.unsqueeze(2) * patches).sum(1)
        ones = previous.new_ones((1, 1, height, width), dtype=torch.float32)
        valid = functional.unfold(ones, kernel_size=kernel, padding=radius)
        valid = valid.reshape(1, kernel * kernel, height, width).bool().expand(batch, -1, -1, -1)
        volume = torch.where(valid, volume, torch.zeros_like(volume))
    return volume, valid


def bidirectional_multiscale(previous: Tensor, current: Tensor) -> tuple[Tensor, Tensor]:
    """100 channels: two directions × two resolutions × 25 offsets."""
    if min(previous.shape[-2:]) < 2:
        raise ValueError("At least 2x2 features required")
    maps, masks = [], []
    for scale in (1, 2):
        a = previous if scale == 1 else functional.avg_pool2d(previous, 2)
        b = current if scale == 1 else functional.avg_pool2d(current, 2)
        for first, second in ((a, b), (b, a)):
            volume, valid = local_cosine_volume(first, second, 2)
            if scale != 1:
                weight = functional.interpolate(
                    valid.float(), size=previous.shape[-2:], mode="bilinear", align_corners=False
                )
                volume = functional.interpolate(
                    volume, size=previous.shape[-2:], mode="bilinear", align_corners=False
                )
                volume = volume / weight.clamp_min(1e-6)
                valid = functional.interpolate(
                    valid.float(), size=previous.shape[-2:], mode="nearest"
                ).bool()
                volume = torch.where(valid, volume, torch.zeros_like(volume))
            maps.append(volume)
            masks.append(valid)
    return torch.cat(maps, 1), torch.cat(masks, 1)


class _ResidualSpatial(nn.Module):
    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class PairMeasurement(nn.Module):
    """Shared encoder, dense pair comparison, then a signed raw-phase readout.

    The output is not a calibrated TTC distribution. Use the native converter in
    the production wrapper and record its clipping/domain explicitly.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        use_correlation: bool = True,
        freeze_encoder: bool = False,
        input_channels: int = 12,
        feature_channels: int = 64,
    ) -> None:
        super().__init__()
        if feature_channels != 64 or input_channels not in (3, 12):
            raise ValueError("Reference recipe requires native 64-D maps and RGB/event channels")
        self.encoder = encoder
        self.use_correlation = use_correlation
        self.freeze_encoder = freeze_encoder
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        # Four dense maps + cosine values + masks + xy coordinates + measured dt.
        in_channels = feature_channels * 4 + 100 + 100 + 2 + 1
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            _ResidualSpatial(),
            _ResidualSpatial(),
        )
        self.readout = nn.Sequential(
            nn.Linear(194, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 1)
        )
        last = self.readout[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.constant_(last.bias, 0.03)
        self.encoder.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()

    @classmethod
    def from_native(
        cls, native: CausalScaleTTC, *, use_correlation: bool = True, freeze_encoder: bool = False
    ) -> PairMeasurement:
        """Copy only the native dense CNN. Does not share or modify its parameters."""
        config = native.config
        return cls(
            copy.deepcopy(native.encoder.features),
            use_correlation=use_correlation,
            freeze_encoder=freeze_encoder,
            input_channels=config.in_channels,
            feature_channels=config.hidden_dim,
        )

    def train(self, mode: bool = True) -> PairMeasurement:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, inputs: Tensor, delta_t_s: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 5 or inputs.shape[1:3] != (2, self.input_channels):
            raise ValueError(
                "P2 requires [B,2,C,H,W], without targets or metadata mixed into pixels"
            )
        batch = inputs.shape[0]
        if delta_t_s.shape not in ((batch,), (batch, 1)):
            raise ValueError("One real time interval per pair required")
        dt = delta_t_s.reshape(batch)
        if not torch.isfinite(dt).all() or not (dt > 0).all():
            raise ValueError("dt must be finite and positive")
        flat = inputs.flatten(0, 1)
        if self.freeze_encoder:
            with torch.no_grad():
                dense = self.encoder(flat)
        else:
            dense = self.encoder(flat)
        dense = dense.reshape(batch, 2, self.feature_channels, *dense.shape[-2:])
        previous, current = dense[:, 0], dense[:, 1]
        if self.use_correlation:
            correlations, support = bidirectional_multiscale(previous, current)
        else:
            # Support depends only on shape, never on scene values or targets.
            ones = previous.new_ones((1, 1, *previous.shape[-2:]))
            with torch.no_grad():
                _, support = bidirectional_multiscale(ones, ones)
            support = support.expand(batch, -1, -1, -1)
            correlations = previous.new_zeros(support.shape)
        height, width = previous.shape[-2:]
        y = (
            torch.arange(height, device=inputs.device, dtype=previous.dtype) + 0.5
        ) / height * 2 - 1
        x = (torch.arange(width, device=inputs.device, dtype=previous.dtype) + 0.5) / width * 2 - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack((xx, yy), 0).unsqueeze(0).expand(batch, -1, -1, -1)
        dt_map = dt.to(previous.dtype).view(batch, 1, 1, 1).expand(-1, -1, height, width)
        pair = torch.cat(
            (
                previous,
                current,
                current - previous,
                current * previous,
                correlations.to(previous.dtype),
                support.to(previous.dtype),
                xy,
                dt_map,
            ),
            dim=1,
        )
        spatial = self.spatial(pair)
        descriptor = torch.cat(
            (
                spatial.mean((-2, -1)),
                spatial.amax((-2, -1)),
                dt[:, None].to(spatial.dtype),
                dt.log()[:, None].to(spatial.dtype),
            ),
            dim=1,
        )
        raw = self.readout(descriptor).squeeze(1)
        return {"raw_phase": raw, "descriptor": descriptor}


def pair_objective(
    raw_phase: Tensor,
    target_phase: Tensor,
    mass: Tensor,
    population: int,
    emitted_phase_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Uniform-sampler estimate: emitted phase L1 + 0.1 raw-phase L1.

    Global masses at sampled rows need not sum to one. This function deliberately
    does NOT multiply by masses again through a weighted sampler.
    """
    if (
        raw_phase.ndim != 1
        or raw_phase.shape != target_phase.shape
        or mass.shape != raw_phase.shape
    ):
        raise ValueError("Aligned per-query scalars required")
    if population < 1 or raw_phase.numel() < 1:
        raise ValueError("Empty objective")
    if not torch.isfinite(target_phase).all() or not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid supervision")
    emitted = emitted_phase_fn(raw_phase)
    per_row = ((emitted - target_phase).abs() + 0.1 * (raw_phase - target_phase).abs()) / 0.03
    return (population * mass * per_row).mean()
