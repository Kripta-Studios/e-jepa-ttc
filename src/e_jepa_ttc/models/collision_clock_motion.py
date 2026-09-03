"""Foreground-free global correspondence features for E-Clock X0."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from e_jepa_ttc.models.local_transport import (
    TRANSPORT_FEATURE_NAMES,
    local_correlation_match,
    transport_physical_features,
)

GLOBAL_TRANSPORT_FEATURE_NAMES: tuple[str, ...] = TRANSPORT_FEATURE_NAMES[:9]


@dataclass(frozen=True)
class GlobalTransportOutput:
    features: torch.Tensor
    dx: torch.Tensor | None
    dy: torch.Tensor | None
    confidence_margin: torch.Tensor | None
    entropy: torch.Tensor | None
    valid: torch.Tensor | None


def height_free_global_transport_features(
    previous_dense: torch.Tensor,
    current_dense: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    return_dense_diagnostics: bool = False,
) -> GlobalTransportOutput:
    """Compute exactly nine uniform global features with forward/reverse matching."""

    forward = local_correlation_match(
        previous_dense,
        current_dense,
        radius=radius,
        temperature=temperature,
        return_probability=False,
    )
    reverse = local_correlation_match(
        current_dense,
        previous_dense,
        radius=radius,
        temperature=temperature,
        return_probability=False,
    )
    features18 = transport_physical_features(
        forward,
        reverse,
        foreground_weight=None,
        radius=radius,
    )
    global9 = features18[..., : len(GLOBAL_TRANSPORT_FEATURE_NAMES)]
    if global9.shape[-1] != 9 or not bool(torch.isfinite(global9).all()):
        raise RuntimeError("global transport feature contract failed")
    if not return_dense_diagnostics:
        return GlobalTransportOutput(global9, None, None, None, None, None)
    return GlobalTransportOutput(
        global9,
        forward.dx,
        forward.dy,
        forward.confidence_margin,
        forward.entropy,
        forward.valid,
    )


__all__ = [
    "GLOBAL_TRANSPORT_FEATURE_NAMES",
    "GlobalTransportOutput",
    "height_free_global_transport_features",
]
