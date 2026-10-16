"""B1 known-texture correspondence objective; no synthesized TTC or event physics."""

from __future__ import annotations

import gc

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from e_jepa_ttc.joint_measurement.model import local_cosine_volume

from .geometry_capture import install_capture, owned_backward
from .geometry_model import GeometricPairMeasurement
from .train import backward

# Each consecutive microbatch has these four transformations, in input pixels.
TRANSFORMS = ((0.9, -2.0, 0.0), (0.9, 2.0, 0.0), (1.1, 0.0, -2.0), (1.1, 0.0, 2.0))


def correspondence_target(size: int, *, reverse: bool) -> tuple[Tensor, Tensor]:
    """Bilinear probability over the native 5x5 displacement grid, on fixed interiors."""
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    # Native stride2 k5/p2 then stride2 k3/p1 have first receptive centre0.5
    # in input pixel-centre coordinates. Pooling2 produces first centre2.5.
    # Both are1.5 input pixels left/up of nominal equal-cell centres.
    offset = 1.5 * size / 128
    x, y = x.float() + 0.5 - size / 2 - offset, y.float() + 0.5 - size / 2 - offset
    probabilities, supports = [], []
    margin = size // 4
    interior = torch.zeros(size, size, dtype=torch.bool)
    interior[margin:-margin, margin:-margin] = True
    offsets = torch.arange(-2, 3)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    for scale, tx, ty in TRANSFORMS:
        tx, ty = tx * size / 128, ty * size / 128
        dx = (x - tx) / scale - x if reverse else (scale - 1) * x + tx
        dy = (y - ty) / scale - y if reverse else (scale - 1) * y + ty
        probability = (1 - (ox.flatten()[:, None, None] - dx).abs()).clamp_min(0)
        probability *= (1 - (oy.flatten()[:, None, None] - dy).abs()).clamp_min(0)
        support = interior & (dx.abs() <= 2) & (dy.abs() <= 2)
        probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-12)
        probabilities.append(probability)
        supports.append(support)
    return torch.stack(probabilities), torch.stack(supports)


def warped_texture(pixels: Tensor) -> Tensor:
    """Transform four P textures with fixed scale/translation, without changing TTC labels."""
    if len(pixels) != 4 or pixels.shape[-2:] != (128, 128):
        raise ValueError("B1 requires four native128 textures")
    theta = pixels.new_tensor(
        [[[1 / s, 0, -tx / 64 / s], [0, 1 / s, -ty / 64 / s]] for s, tx, ty in TRANSFORMS]
    )
    grid = functional.affine_grid(theta, list(pixels.shape), align_corners=False)
    return functional.grid_sample(pixels, grid, align_corners=False, padding_mode="zeros")


class CorrespondenceCore(nn.Module):
    """Shared encoder, four native cosine volumes and fixed bilinear correspondence loss."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        for size in (32, 16):
            for reverse in (False, True):
                target, support = correspondence_target(size, reverse=reverse)
                self.register_buffer(f"target_{size}_{int(reverse)}", target)
                self.register_buffer(f"support_{size}_{int(reverse)}", support)

    def forward(self, pixels: Tensor) -> Tensor:
        """Mean CE over supported locations, directions, resolutions and four queries."""
        pair = torch.stack((pixels, warped_texture(pixels)), 1)
        maps = self.encoder(pair.flatten(0, 1)).reshape(4, 2, 64, 32, 32)
        loss = pixels.new_zeros(())
        for size in (32, 16):
            first, second = maps[:, 0], maps[:, 1]
            if size == 16:
                first, second = functional.avg_pool2d(first, 2), functional.avg_pool2d(second, 2)
            for reverse in (False, True):
                a, b = (second, first) if reverse else (first, second)
                volume, valid = local_cosine_volume(a, b)
                log_probability = (volume / 0.07).masked_fill(~valid, -1e4).log_softmax(1)
                target = getattr(self, f"target_{size}_{int(reverse)}")
                support = getattr(self, f"support_{size}_{int(reverse)}")
                ce = -(target * log_probability).sum(1)
                loss = loss + ((ce * support).sum((1, 2)) / support.sum((1, 2))).mean() / 4
        return loss


def step_backward(
    model: GeometricPairMeasurement,
    batches: list,
    weights: Tensor,
    population: int,
    *,
    correspondence: bool,
    capture: bool,
) -> dict:
    """Keep graph-owned gradients separate; initialize graphs only after complete restore."""
    first = batches[0]
    if correspondence:
        core = getattr(model, "_b1_correspondence", None)
        if core is None:
            core = CorrespondenceCore(model.encoder).to(first.events.device)
            if capture:
                gc.collect()
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                try:
                    torch.cuda.make_graphed_callables(
                        core, (first.events[:, -1].detach().clone(),), num_warmup_iters=3
                    )
                finally:
                    torch.set_rng_state(cpu)
                    torch.cuda.set_rng_state_all(cuda)
            object.__setattr__(model, "_b1_correspondence", core)
        for parameter in model.encoder.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        total = 0.0
        for batch in batches:
            loss = core(batch.events[:, -1]) / len(batches)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite B1 correspondence loss")
            loss.backward()
            total += float(loss.detach())
        return {"loss": total}
    if capture:
        if not getattr(model, "_v18_capture_active", False):
            gc.collect()
            install_capture(model, first.events, first.delta_t_s.reshape(len(first.events)))
        return owned_backward(model, batches, weights, population)
    return backward(model, batches, weights, population)
