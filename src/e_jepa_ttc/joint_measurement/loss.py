"""Prospective P-only masses and the canonical emitted/raw phase loss."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import emitted_phase


def training_mass(target: np.ndarray, groups: np.ndarray, *, role: str) -> np.ndarray:
    """Equal group mass and historical band proportions, fitted only on P."""
    if role != "P":
        raise ValueError("Only P may fit producer training masses")
    target, groups = np.asarray(target, dtype=np.float64), np.asarray(groups).astype(str)
    if target.ndim != 1 or target.shape != groups.shape or not len(target):
        raise ValueError("Aligned nonempty targets and groups required")
    masks = [
        (target > 0.1) & (target <= 3),
        (target > 3) & (target <= 6),
        (target > 6) & (target <= 10),
        (target > -10) & (target < 0),
    ]
    if not np.isfinite(target).all() or not np.logical_or.reduce(masks).all():
        raise ValueError("Out-of-domain P target: no silent row deletion")
    names = np.unique(groups)
    mass = np.zeros(len(target), dtype=np.float64)
    for group in names:
        present = [
            (mask & (groups == group), weight)
            for mask, weight in zip(masks, (0.5, 0.3, 0.1, 0.1), strict=True)
            if (mask & (groups == group)).any()
        ]
        total = sum(weight for _, weight in present)
        for mask, weight in present:
            mass[mask] = weight / total / len(names) / int(mask.sum())
    if not np.isclose(mass.sum(), 1, atol=1e-12):
        raise ValueError("Training mass is not normalized")
    return mass


def phase_components(raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return unweighted per-query terms to retain exact effective-batch scaling."""
    if raw.ndim != 1 or raw.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError("Aligned finite phase targets required")
    emitted = emitted_phase(raw)
    return {
        "emitted": (emitted - target).abs() / 0.03,
        "raw": 0.1 * (raw - target).abs() / 0.03,
        "raw_emitted_gap": (raw - emitted).abs(),
    }
