"""RGB-anchored phase complement with an equal-capacity event-ZERO control.

This module does not admit parents, features, normalizers or a training recipe.
Those identities must be registered together before any scientific update.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, ttc_to_phase


class NativePhaseConversion(nn.Module):
    """Canonical conversion with constants stored on the module for TorchScript.

    The shared function closes over Python floats, which TorchScript cannot
    compile. These operations and bounds are checked against that function,
    including signed saturation, zero phase and derivatives.
    """

    __constants__ = ["minimum", "maximum"]

    def __init__(self) -> None:
        super().__init__()
        self.minimum = MIN_PHASE
        self.maximum = MAX_PHASE

    def forward(self, phase: Tensor) -> Tensor:
        """Preserve the native signed output domain and 60-second saturation."""
        if not torch.isfinite(phase).all():
            raise ValueError("nonfinite phase")
        phi = phase.clamp(self.minimum, self.maximum)
        q = -torch.expm1(-phi) / 0.1
        sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
        return sign / q.abs().clamp_min(1.0 / 60.0)


class RGBAnchorComplement(nn.Module):
    """Learn a phase correction to RGB; copy event TTC when RGB is missing.

    Features include any separately admitted sensor-clock coordinates. ZERO
    removes the entire event feature block, including those coordinates; event
    TTC is used exclusively for missing-RGB fallback. No TTC averaging occurs.
    Initial correction is zero, preserving RGB up to native phase roundoff.
    """

    def __init__(
        self,
        rgb_features: int,
        event_features: int,
        *,
        hidden: int = 128,
        residual_scale: float = 0.03,
        mode: str = "TRUE",
    ) -> None:
        super().__init__()
        if min(rgb_features, event_features, hidden) < 1:
            raise ValueError("Positive feature and hidden dimensions required")
        if mode not in {"TRUE", "ZERO"} or not 0 < residual_scale <= 1:
            raise ValueError("Declare TRUE/ZERO and a finite residual scale in (0,1]")
        self.rgb_features = rgb_features
        self.event_features = event_features
        self.mode = mode
        self.residual_scale = residual_scale
        self.convert = NativePhaseConversion()
        self.network = nn.Sequential(
            nn.Linear(rgb_features + event_features + 1, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        rgb_features: Tensor,
        event_features: Tensor,
        rgb_ttc: Tensor,
        event_ttc: Tensor,
        rgb_available: Tensor,
    ) -> dict[str, Tensor]:
        """Accept missing RGB values only in rows that take the exact event fallback."""
        batch = len(event_ttc)
        if (
            event_ttc.shape != (batch,)
            or rgb_ttc.shape != (batch,)
            or rgb_features.shape != (batch, self.rgb_features)
            or event_features.shape != (batch, self.event_features)
            or rgb_available.shape != (batch,)
            or rgb_available.dtype != torch.bool
            or batch < 1
        ):
            raise ValueError("Fusion requires aligned nonempty features, TTC and availability")
        tensors = (rgb_features, event_features, rgb_ttc, event_ttc, rgb_available)
        if any(value.device != event_ttc.device for value in tensors):
            raise ValueError("Fusion inputs must share a device")
        if any(value.dtype != torch.float32 for value in tensors[:-1]):
            raise ValueError("Fusion input contract requires FP32")
        event_phase = ttc_to_phase(event_ttc)
        result = {
            "raw_phase": event_phase.clone(),
            "point_phase": event_phase.clone(),
            "ttc": event_ttc.clone(),
            "used_rgb": rgb_available.clone(),
        }
        if not rgb_available.any():
            return result
        selected = rgb_available.nonzero().squeeze(1)
        rgb = rgb_features[selected]
        event = event_features[selected]
        if self.mode == "ZERO":
            event = torch.zeros_like(event)
        if not torch.isfinite(rgb).all() or not torch.isfinite(event).all():
            raise ValueError("Available fusion features must be finite")
        anchor = ttc_to_phase(rgb_ttc[selected])
        hidden = torch.cat((rgb, event, anchor[:, None]), dim=1)
        raw = anchor + self.residual_scale * self.network(hidden).squeeze(1)
        ttc = self.convert(raw)
        result["raw_phase"][selected] = raw
        result["point_phase"][selected] = ttc_to_phase(ttc)
        result["ttc"][selected] = ttc
        return result
