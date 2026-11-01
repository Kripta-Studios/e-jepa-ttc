"""Frozen PAIR at canonical cadence with explicit raw-phase interval conversion."""

import torch
from torch import nn

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.features import pair133


class CanonicalPhaseDomainError(ValueError):
    """Physical raw-phase conversion left its real finite domain; inputs were validated."""


class CanonicalPair(nn.Module):
    """Preserve original full batches and leave emitted_phase solely to the existing caller.

    Identity rows bypass phase conversion, even when converting their finite raw
    phase would overflow FP32. Diagnostic inverses may therefore be infinite on
    bypassed identity rows; this never changes their original phase bytes.
    """

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.original = original
        self.last_trace: dict[str, torch.Tensor] = {}

    @staticmethod
    def _phase(value: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (len(features),)
            or value.dtype != torch.float32
            or value.device != features.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("PAIR must return finite FP32[B] raw phase on its input device")
        return value

    def forward(self, batch: PairFeatureBatch) -> torch.Tensor:
        """Canonicalize all three time columns and convert before any emission policy."""
        self.last_trace = {}
        if not isinstance(batch, PairFeatureBatch):
            raise ValueError("Canonical PAIR requires inference-only PairFeatureBatch")
        features = batch.features
        if (
            features.ndim != 2
            or features.shape[1] != 133
            or len(features) == 0
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Canonical PAIR requires nonempty finite FP32[B,133]")
        dt, support = features[:, 128], features[:, 131:133]
        if (dt <= 0).any() or ((support < 0) | (support > 1)).any():
            raise ValueError("Canonical PAIR requires positive dt and supports in[0,1]")
        d0 = features.new_tensor(0.1)
        identity = dt == d0
        original = self._phase(self.original(batch), features)
        canonical = original
        if not identity.all():
            canonical_features = pair133(features[:, :128], torch.full_like(dt, 0.1), support)
            canonical = self._phase(self.original(PairFeatureBatch(canonical_features)), features)
        # Raw conversion deliberately excludes phase_to_ttc and its clipping/magnitude floor.
        canonical_q = -torch.expm1(-canonical) / d0
        original_q = -torch.expm1(-original) / d0
        corrected_q = torch.where(identity, original_q, canonical_q * d0 / dt)
        domain_valid = identity | (torch.isfinite(corrected_q) & ((1 - d0 * corrected_q) > 0))
        corrected = original.clone()
        active = ~identity & domain_valid
        corrected[active] = -torch.log1p(-d0 * corrected_q[active])
        corrected[~domain_valid] = float("nan")
        self.last_trace = {
            key: value.detach().clone()
            for key, value in {
                "original_raw_phase": original,
                "canonical_raw_phase": canonical,
                "canonical_q": canonical_q,
                "corrected_q": corrected_q,
                "raw_corrected_phase": corrected,
                "identity_mask": identity,
                "real_dt": dt,
                "domain_valid": domain_valid,
            }.items()
        }
        if not domain_valid.all() or not torch.isfinite(corrected).all():
            raise CanonicalPhaseDomainError(
                "Canonical PAIR correction outside finite phase domain:1-d0*q<=0 or overflow"
            )
        return corrected
