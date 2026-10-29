"""Candidate CUDA loss graph; identical phase arithmetic, no scientific admission implied."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .geometry_capture import install_capture
from .geometry_model import GeometricPairMeasurement
from .phase_transport_loss import phase_terms


class LossCore(nn.Module):
    """Parameterless ordered loss and detached diagnostics for one microbatch."""

    def __init__(self, population: int, batch_size: int) -> None:
        super().__init__()
        if population <= 0 or batch_size <= 0:
            raise ValueError("Positive population and full batch size required")
        self.population = population
        self.batch_size = batch_size

    def forward(self, raw: Tensor, truth: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Keep the existing operator and reduction order, including the raw auxiliary term."""
        if raw.shape != weights.shape:
            raise ValueError("One original mass per microbatch query required")
        terms, valid = phase_terms(raw, truth)
        scaled = self.population * weights
        loss = ((terms["emitted"] + terms["raw"]) * scaled).sum() / self.batch_size
        diagnostics = torch.stack(
            [
                loss.detach(),
                *[(value.detach() * scaled).sum() / self.batch_size for value in terms.values()],
            ]
        )
        return loss, diagnostics, valid & torch.isfinite(loss)


def checked(result: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Check the domain before backward; own diagnostic storage before the next replay."""
    loss, diagnostics, valid = result
    if not bool(valid):
        raise FloatingPointError("Original phase domain/finite loss guard failed")
    return loss, diagnostics.detach().clone()


def signature(values: tuple[Tensor, Tensor, Tensor]) -> tuple:
    """Bind shape, dtype, device and differentiability without synchronizing CUDA."""
    return tuple((tuple(x.shape), x.dtype, x.device, x.requires_grad) for x in values)


class CapturedLoss:
    """One private forward/backward graph, retained across microbatches and scopes."""

    def __init__(
        self, raw: Tensor, truth: Tensor, weights: Tensor, population: int, batch_size: int
    ) -> None:
        args = (raw, truth, weights)
        if (
            any(x.device != raw.device or x.dtype != torch.float32 for x in args)
            or raw.device.type != "cuda"
            or not raw.requires_grad
            or truth.requires_grad
            or weights.requires_grad
        ):
            raise ValueError("CUDA FP32 phase gradients and nontrainable targets/masses required")
        self.signature = signature(args)
        self.population = population
        self.batch_size = batch_size
        self.core = LossCore(population, batch_size)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        try:
            # Never use the existing model graph's output storage as a static input.
            samples = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in args)
            torch.cuda.make_graphed_callables(self.core, samples, num_warmup_iters=3)
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        self.capture_count = 1

    def __call__(self, raw: Tensor, truth: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        """Reject graph-contract drift and preserve the pre-backward finite/domain check."""
        if signature((raw, truth, weights)) != self.signature:
            raise ValueError("Captured loss input signature changed")
        return checked(self.core(raw, truth, weights))


def backward(
    model: GeometricPairMeasurement, batches: list, weights: Tensor, population: int
) -> dict:
    """Retain eight ordered backward calls and exact Python-double diagnostic aggregation."""
    if not batches or weights is None:
        raise ValueError("Nonempty microbatches and original population masses required")
    first = batches[0]
    if not getattr(model, "_v18_capture_active", False):
        install_capture(model, first.events, first.delta_t_s.reshape(len(first.events)))
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    size = sum(len(batch.events) for batch in batches)
    if weights.shape != (size,):
        raise ValueError("One mass per sampled query required")
    start, diagnostics = 0, []
    for batch in batches:
        n = len(batch.events)
        raw = model(batch.events, batch.delta_t_s.reshape(n))["raw_phase"]
        masses = weights[start : start + n].to(raw.device)
        core = getattr(model, "_v18_phase_loss", None)
        if core is None:
            core = CapturedLoss(raw, batch.target_ttc_s, masses, population, size)
            # Ephemeral runtime object: no parameters, buffers or checkpoint keys added.
            object.__setattr__(model, "_v18_phase_loss", core)
        if core.population != population or core.batch_size != size:
            raise ValueError("Captured loss population or full batch size changed")
        loss, diagnostic = core(raw, batch.target_ttc_s, masses)
        loss.backward()
        diagnostics.append(diagnostic)
        start += n
    names = ("loss", "emitted", "raw", "raw_emitted_gap")
    result = dict.fromkeys(names, 0.0)
    for values in torch.stack(diagnostics).cpu().tolist():
        for name, value in zip(names, values, strict=True):
            result[name] += value
    return result
