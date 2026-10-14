"""C1 tensor-only capture candidate; original scientific model remains unchanged.

Arithmetic matches geometry_model. Dynamic guards are returned as one boolean
tensor and checked before the caller can backpropagate or update any weight.
This module alone does not admit a scientific runtime transition.
"""

from __future__ import annotations

import math
from dataclasses import fields

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.joint_measurement.model import bidirectional_multiscale

from .geometry_model import GeometricPairMeasurement


def contiguous_microbatch(batch: ObjectEventV4Batch, start: int, stop: int) -> ObjectEventV4Batch:
    """Preserve pinned storage, values and identities for a contiguous microbatch."""
    if not 0 <= start < stop <= len(batch.events):
        raise ValueError("Invalid contiguous microbatch bounds")
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = value[start:stop] if isinstance(value, (Tensor, list)) else None
    return ObjectEventV4Batch(**values)


def affine_tensor(flow: Tensor, weights: Tensor, ridge: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Same ordered affine operations; return the original dynamic guard separately."""
    valid = torch.isfinite(flow).all() & torch.isfinite(weights).all() & ~(weights < 0).any()
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
    return matrix, valid


class CaptureCore(nn.Module):
    """Share every original parameter; expose only the phase consumed by training."""

    def __init__(self, model: GeometricPairMeasurement) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.spatial = model.spatial
        self.gates = model.gates
        self.spatial_weights = model.spatial_weights
        self.input_channels = model.input_channels
        self.temperature = model.temperature
        self.train(model.training)

    def forward(self, inputs: Tensor, delta_t_s: Tensor) -> tuple[Tensor, Tensor]:
        """Keep FP32 operations and dropout order, delaying only boolean materialization."""
        if inputs.ndim != 5 or inputs.shape[1:] != (2, self.input_channels, 128, 128):
            raise ValueError("Geometry recipe requires the native 128x128 pair")
        batch = len(inputs)
        if delta_t_s.shape not in {(batch,), (batch, 1)}:
            raise ValueError("One native interval per query required")
        dt = delta_t_s.reshape(batch)
        valid_all = torch.isfinite(dt).all() & (dt > 0).all()
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
        scales = []
        for index, scale in enumerate((1, 1, 2, 2)):
            volume = correlations[:, index * 25 : (index + 1) * 25]
            valid = support[:, index * 25 : (index + 1) * 25]
            probability = (volume / self.temperature).masked_fill(~valid, -torch.inf).softmax(1)
            dx = (probability * ox.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            dy = (probability * oy.reshape(1, 25, 1, 1)).sum(1) * (scale / 16)
            interior = torch.zeros_like(weights[:, index])
            margin = 2 * scale
            interior[:, margin:-margin, margin:-margin] = 1
            matrix, finite = affine_tensor(torch.stack((dx, dy), 1), weights[:, index] * interior)
            valid_all = valid_all & finite
            log_axes = 0.5 * matrix.square().sum(1).clamp_min(1e-12).log()
            scales.append(gates[:, 0] * log_axes[:, 0] + (1 - gates[:, 0]) * log_axes[:, 1])
        fine = 0.5 * (scales[0] - scales[1])
        coarse = 0.5 * (scales[2] - scales[3])
        ratio = (1 - gates[:, 1]) * fine + gates[:, 1] * coarse
        valid_all = valid_all & torch.isfinite(ratio).all()
        inverse_ttc = torch.expm1(ratio) / dt
        safe_inverse = inverse_ttc.clamp(max=9.9)
        raw = torch.where(
            inverse_ttc > 9.9,
            -math.log(0.01) + (inverse_ttc - 9.9) * 10,
            -torch.log1p(-0.1 * safe_inverse),
        )
        return raw, valid_all


def checked_phase(result: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    """Materialize the captured guards before returning a phase for differentiation."""
    raw, valid = result
    if not bool(valid):
        raise ValueError("Geometry finite/domain guard failed")
    return {"raw_phase": raw}


def install_capture(model: GeometricPairMeasurement, inputs: Tensor, dt: Tensor) -> CaptureCore:
    """Capture forward and backward together; preserve original weights and RNG state."""
    if inputs.device.type != "cuda" or not model.training:
        raise ValueError("Capture requires CUDA training samples")
    core = CaptureCore(model)
    ids = {key: id(value) for key, value in model.named_parameters()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        torch.cuda.make_graphed_callables(core, (inputs, dt), num_warmup_iters=3)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    if ids != {key: id(value) for key, value in core.named_parameters()}:
        raise ValueError("Capture changed optimizer parameter identities")

    def forward(new_inputs: Tensor, new_dt: Tensor) -> dict[str, Tensor]:
        if new_inputs.shape != inputs.shape or new_dt.shape != dt.shape:
            raise ValueError("Captured microbatch shape changed")
        return checked_phase(core(new_inputs, new_dt))

    model.forward = forward
    return core
