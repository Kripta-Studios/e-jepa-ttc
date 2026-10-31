"""Small quality-conditioned convex fusion with an exact native fallback."""

import numpy as np
import torch
from torch import nn


def fit_normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit robust scales on explicitly supplied training observations only."""
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Normalization requires a nonempty finite training matrix")
    center = np.median(x, axis=0)
    scale = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
    return center.astype(np.float32), np.maximum(scale, 0.05).astype(np.float32)


class QualityFusion(nn.Module):
    """Blend signed native TTC and causal geometry without crossing their range.

    Both controls use the geometric expert in the output. The native control
    removes quality from the gate, not from the estimator. Its wider layer keeps
    the same 401 trainable parameters. No sequence or target information enters.
    """

    center: torch.Tensor
    scale: torch.Tensor

    def __init__(self, mode: str = "quality") -> None:
        super().__init__()
        if mode not in ("quality", "native"):
            raise ValueError("Expected quality or native gate")
        self.quality_mode = mode == "quality"
        dim, hidden = (23, 16) if self.quality_mode else (3, 80)
        self.register_buffer("center", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        input_layer = nn.Linear(dim, hidden)
        output_layer = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(input_layer, nn.Tanh(), output_layer)
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, -2.0)

    @torch.jit.export
    def inputs(self, native: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        safe = torch.nan_to_num(native, nan=0.0, posinf=60.0, neginf=-60.0)
        signed_log = torch.sign(safe) * torch.log1p(torch.abs(safe))
        if self.quality_mode:
            return torch.cat(
                (signed_log, torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)), 1
            )
        return signed_log

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """Return seconds, preserving NaN only when both sources are absent."""
        base = native[:, 0]
        base_ok = torch.isfinite(base)
        geo_ok = torch.isfinite(geometry) & torch.isfinite(quality).all(1)
        geo = torch.nan_to_num(geometry, nan=0.0, posinf=60.0, neginf=-60.0).clamp(-60.0, 60.0)
        safe_base = torch.where(base_ok, base, geo)
        x = ((self.inputs(native, quality) - self.center) / self.scale).clamp(-8.0, 8.0)
        weight = torch.sigmoid(self.gate(x).squeeze(1))
        compatible = geo_ok & ((torch.sign(geo) == torch.sign(safe_base)) | ~base_ok)
        weight = torch.where(compatible, weight, torch.zeros_like(weight))
        mixed = safe_base + weight * (geo - safe_base)
        return torch.where(base_ok, mixed, torch.where(geo_ok, geo, base))
