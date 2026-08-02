"""Deterministic token and attention-cost calculations for high-resolution runs."""

from __future__ import annotations

import math


def patch_grid_tokens(height: int, width: int, patch_size: int) -> int:
    """Return ceil-padded spatial patch count without dropping borders."""

    if min(height, width, patch_size) <= 0:
        raise ValueError("height, width and patch_size must be positive.")
    return math.ceil(height / patch_size) * math.ceil(width / patch_size)


def global_attention_pairs(steps: int, patches: int) -> int:
    """Return the score-matrix pair count for global attention."""

    if min(steps, patches) <= 0:
        raise ValueError("steps and patches must be positive.")
    return (steps * patches) ** 2


def temporal_factorized_pairs(steps: int, patches: int) -> int:
    """Return the score-pair count when temporal attention is per patch."""

    if min(steps, patches) <= 0:
        raise ValueError("steps and patches must be positive.")
    return patches * steps**2


__all__ = ["global_attention_pairs", "patch_grid_tokens", "temporal_factorized_pairs"]
