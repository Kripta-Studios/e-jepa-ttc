"""Apparent-area TTC expert."""

from __future__ import annotations

import torch

from e_jepa_ttc.geometry.height_ratio_ttc import _causal_pair_ratio_rate


def area_rate_inverse_ttc(
    areas: torch.Tensor,
    times_s: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate current-endpoint inverse TTC from square-root area growth."""

    return _causal_pair_ratio_rate(
        areas,
        times_s,
        valid_mask=valid_mask,
        ratio_power=0.5,
    )


__all__ = ["area_rate_inverse_ttc"]
