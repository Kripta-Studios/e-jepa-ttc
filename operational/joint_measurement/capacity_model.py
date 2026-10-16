"""Family D: twice-wide pair predictor, retaining the trainable native CNN."""

from __future__ import annotations

import copy
import gc

import torch
from torch import Tensor, nn

from e_jepa_ttc.joint_measurement.model import (
    PairMeasurement,
    _ResidualSpatial,
    bidirectional_multiscale,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

from .geometry_capture import checked_phase, owned_backward


class WidePairMeasurement(PairMeasurement):
    """Double spatial/readout widths only; preserve native pretrained encoder ancestry.

    The encoder remains 64 channels and is fully trainable. This is a predictor
    capacity contrast, not a claim that every dimension of the system doubled.
    Width one reproduces the original constructor and its initialization order.
    """

    def __init__(self, parent: CausalScaleTTC, *, width: int = 2) -> None:
        nn.Module.__init__(self)
        if width not in (1, 2) or parent.config.hidden_dim != 64:
            raise ValueError("Only the matched native64 predictor widths one/two are admitted")
        self.encoder = copy.deepcopy(parent.encoder.features)
        self.use_correlation = True
        self.freeze_encoder = False
        self.input_channels = parent.config.in_channels
        self.feature_channels = 64
        self.width = width
        channels, hidden = 96 * width, 128 * width
        self.spatial = nn.Sequential(
            nn.Conv2d(459, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ResidualSpatial(channels),
            _ResidualSpatial(channels),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * channels + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        last = self.readout[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.constant_(last.bias, 0.03)
        self.encoder.requires_grad_(True)


class PairTensorCore(nn.Module):
    """Original ordered pair arithmetic with tensor-valued native-time guards."""

    def __init__(self, model: WidePairMeasurement) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.spatial = model.spatial
        self.readout = model.readout
        self.input_channels = model.input_channels
        self.train(model.training)

    def forward(self, inputs: Tensor, delta_t_s: Tensor) -> tuple[Tensor, Tensor]:
        """No labels or cached neural features enter this differentiable forward."""
        if inputs.ndim != 5 or inputs.shape[1:] != (2, self.input_channels, 128, 128):
            raise ValueError("Native P2 pixels required")
        batch = inputs.shape[0]
        if delta_t_s.shape not in ((batch,), (batch, 1)):
            raise ValueError("One native interval per pair required")
        dt = delta_t_s.reshape(batch)
        valid = torch.isfinite(dt).all() & (dt > 0).all()
        dense = self.encoder(inputs.flatten(0, 1)).reshape(batch, 2, 64, 32, 32)
        previous, current = dense[:, 0], dense[:, 1]
        correlations, support = bidirectional_multiscale(previous, current)
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
        return self.readout(descriptor).squeeze(1), valid


def install(model: WidePairMeasurement, inputs: Tensor, delta: Tensor) -> None:
    """Capture before allocating owned gradient buffers, preserving RNG and parameters."""
    if inputs.device.type != "cuda" or not model.training:
        raise ValueError("CUDA training capture required")
    # Collect prior unreachable graph cycles outside a capture boundary.
    gc.collect()
    core = PairTensorCore(model)
    identities = {name: id(value) for name, value in model.named_parameters()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        torch.cuda.make_graphed_callables(
            core, (inputs.detach().clone(), delta.detach().clone()), num_warmup_iters=3
        )
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    if identities != {name: id(value) for name, value in core.named_parameters()}:
        raise ValueError("Capture changed optimizer parameter identities")
    input_shape, time_shape = inputs.shape, delta.shape

    def forward(inputs: Tensor, delta_t_s: Tensor) -> dict[str, Tensor]:
        if inputs.shape != input_shape or delta_t_s.shape != time_shape:
            raise ValueError("Captured input shape changed")
        return checked_phase(core(inputs, delta_t_s))

    model.forward = forward
    object.__setattr__(model, "_v18_capacity_captured", True)


def captured_backward(
    model: WidePairMeasurement,
    batches: list,
    weights: Tensor,
    population: int,
    *,
    recompute: bool = False,
) -> dict:
    """Build the graph before grad-slot initialization; then use proven owned accumulation."""
    if not getattr(model, "_v18_capacity_captured", False):
        first = batches[0]
        install(model, first.events, first.delta_t_s.reshape(len(first.events)))
    return owned_backward(model, batches, weights, population, recompute=recompute)
