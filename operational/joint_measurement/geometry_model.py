"""Candidate family-C geometry readout; no admitted training recipe or results yet.

The encoder and common spatial tower remain trainable. A weighted affine fit
reads expected displacements from the same 100-channel native cosine volume.
Two learned gates mix axes and resolutions; there is no free TTC residual.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from e_jepa_ttc.joint_measurement.model import PairMeasurement, bidirectional_multiscale
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC


def weighted_affine(flow: Tensor, weights: Tensor, ridge: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Fit current = A @ previous + translation on normalized pixel-centre coordinates.

    Flow is [B,2,H,W] in those same coordinates. Positive ridge supplies a finite
    identity-map fallback for zero support; no minimum number of matches gates
    prediction. Translation is estimated jointly, not treated as expansion.
    """
    if flow.ndim != 4 or flow.shape[1] != 2 or weights.shape != flow[:, 0].shape:
        raise ValueError("Affine flow/weight shapes differ")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("Positive finite affine ridge required")
    if not torch.isfinite(flow).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Affine inputs must be finite with nonnegative weights")
    batch, _, height, width = flow.shape
    y = (torch.arange(height, device=flow.device, dtype=flow.dtype) + 0.5) * (2 / height) - 1
    x = (torch.arange(width, device=flow.device, dtype=flow.dtype) + 0.5) * (2 / width) - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx, yy))[None].expand(batch, -1, -1, -1)
    normalized = weights / weights.sum((-2, -1), keepdim=True).clamp_min(1e-12)
    center = (coordinates * normalized[:, None]).sum((-2, -1))
    motion = (flow * normalized[:, None]).sum((-2, -1))
    position = coordinates - center[:, :, None, None]
    residual = flow - motion[:, :, None, None]
    px, py = position.unbind(1)
    cxx = (normalized * px.square()).sum((-2, -1)) + ridge
    cyy = (normalized * py.square()).sum((-2, -1)) + ridge
    cxy = (normalized * px * py).sum((-2, -1))
    dx = (residual * (normalized * px)[:, None]).sum((-2, -1))
    dy = (residual * (normalized * py)[:, None]).sum((-2, -1))
    determinant = (cxx * cyy - cxy.square()).clamp_min(ridge * ridge)
    bx = (dx * cyy[:, None] - dy * cxy[:, None]) / determinant[:, None]
    by = (dy * cxx[:, None] - dx * cxy[:, None]) / determinant[:, None]
    slope = torch.stack((bx, by), dim=-1)
    matrix = slope + torch.eye(2, dtype=flow.dtype, device=flow.device)[None]
    translation = motion - torch.einsum("bij,bj->bi", slope, center)
    return matrix, translation


def relative_scale_phase(log_ratio: Tensor, delta_t_s: Tensor) -> Tensor:
    """Map previous-to-current size ratio to TTC at the CURRENT instant.

    Under constant closure ratio=1+dt/T_current. Native phase is -log(1-.1/T).
    Extend its tangent beyond inverse TTC 9.9 to retain training gradients;
    the existing native phase-to-TTC converter still controls emitted support.
    """
    if log_ratio.shape != delta_t_s.shape or not (delta_t_s > 0).all():
        raise ValueError("Relative scale requires aligned positive native time intervals")
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(delta_t_s).all():
        raise ValueError("Relative scale/time must be finite")
    inverse_ttc = torch.expm1(log_ratio) / delta_t_s
    safe_inverse = inverse_ttc.clamp(max=9.9)
    return torch.where(
        inverse_ttc > 9.9,
        -math.log(0.01) + (inverse_ttc - 9.9) * 10,
        -torch.log1p(-0.1 * safe_inverse),
    )


class GeometricPairMeasurement(nn.Module):
    """Private native CNN and shared pair tower with explicit affine scale readout."""

    def __init__(self, parent: CausalScaleTTC, *, temperature: float = 0.02) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive finite matching temperature required")
        pair = PairMeasurement.from_native(parent, use_correlation=True, freeze_encoder=False)
        self.encoder = pair.encoder
        self.spatial = pair.spatial
        self.gates = nn.Sequential(*list(pair.readout.children())[:-1], nn.Linear(128, 2))
        self.spatial_weights = nn.Conv2d(96, 4, 1)
        self.input_channels = pair.input_channels
        self.temperature = temperature
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.zeros_(self.gates[-1].bias)
        nn.init.zeros_(self.spatial_weights.weight)
        nn.init.zeros_(self.spatial_weights.bias)

    def forward(self, inputs: Tensor, delta_t_s: Tensor) -> dict[str, Tensor]:
        """Use only native pixels and native time; no boxes, labels or expert TTC."""
        if inputs.ndim != 5 or inputs.shape[1:] != (2, self.input_channels, 128, 128):
            raise ValueError("Geometry recipe requires the native 128x128 pair")
        batch = len(inputs)
        if delta_t_s.shape not in {(batch,), (batch, 1)}:
            raise ValueError("One native interval per query required")
        dt = delta_t_s.reshape(batch)
        if not torch.isfinite(dt).all() or not (dt > 0).all():
            raise ValueError("Native time intervals must be positive and finite")
        dense = self.encoder(inputs.flatten(0, 1)).reshape(batch, 2, 64, 32, 32)
        previous, current = dense[:, 0], dense[:, 1]
        correlations, support = bidirectional_multiscale(previous, current)
        axis = (torch.arange(32, dtype=inputs.dtype, device=inputs.device) + 0.5) / 16 - 1
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((xx, yy))[None].expand(batch, -1, -1, -1)
        spatial = self.spatial(
            torch.cat(
                (
                    previous,
                    current,
                    current - previous,
                    current * previous,
                    correlations,
                    support.to(previous.dtype),
                    coordinates,
                    dt[:, None, None, None].expand(-1, 1, 32, 32),
                ),
                dim=1,
            )
        )
        descriptor = torch.cat(
            (spatial.mean((-2, -1)), spatial.amax((-2, -1)), dt[:, None], dt.log()[:, None]), dim=1
        )
        gates = self.gates(descriptor).sigmoid()
        weights = functional.softplus(self.spatial_weights(spatial)) + 1e-4
        offsets = torch.arange(-2, 3, dtype=inputs.dtype, device=inputs.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        matrices, translations, scales = [], [], []
        for index, scale in enumerate((1, 1, 2, 2)):
            volume = correlations[:, index * 25 : (index + 1) * 25]
            valid = support[:, index * 25 : (index + 1) * 25]
            probability = (volume / self.temperature).masked_fill(~valid, -torch.inf).softmax(1)
            dx = (probability * ox.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            dy = (probability * oy.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            interior = torch.zeros_like(weights[:, index])
            margin = 2 * scale
            interior[:, margin:-margin, margin:-margin] = 1
            matrix, translation = weighted_affine(
                torch.stack((dx, dy), 1), weights[:, index] * interior
            )
            log_axes = 0.5 * matrix.square().sum(1).clamp_min(1e-12).log()
            scales.append(gates[:, 0] * log_axes[:, 0] + (1 - gates[:, 0]) * log_axes[:, 1])
            matrices.append(matrix)
            translations.append(translation)
        fine = 0.5 * (scales[0] - scales[1])
        coarse = 0.5 * (scales[2] - scales[3])
        ratio = (1 - gates[:, 1]) * fine + gates[:, 1] * coarse
        return {
            "raw_phase": relative_scale_phase(ratio, dt),
            "descriptor": descriptor,
            "log_relative_scale": ratio,
            "axis_scale_gates": gates,
            "affine_matrices": torch.stack(matrices, 1),
            "translations": torch.stack(translations, 1),
        }
