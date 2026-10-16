"""Prospective group/band masses; the native emitted/raw objective is unchanged."""

from __future__ import annotations

import numpy as np

MASSES = {"L0": (0.5, 0.3, 0.1, 0.1), "LW": (4 / 9, 4 / 15, 4 / 45, 1 / 5)}


def training_mass(target: np.ndarray, groups: np.ndarray, arm: str) -> np.ndarray:
    """Keep uniform sampling and assign equal group mass, never minibatch masses."""
    from e_jepa_ttc.joint_measurement.loss import training_mass as native

    if arm not in MASSES:
        raise ValueError("Only the registered L0/LW contrast is supported")
    reference = native(target, groups, role="P")
    if arm == "L0":
        return reference
    target, groups = np.asarray(target, dtype=np.float64), np.asarray(groups).astype(str)
    masks = (
        (target > 0.1) & (target <= 3),
        (target > 3) & (target <= 6),
        (target > 6) & (target <= 10),
        (target > -10) & (target < 0),
    )
    names = np.unique(groups)
    result = np.zeros_like(reference)
    for group in names:
        present = [
            (mask & (groups == group), mass)
            for mask, mass in zip(masks, MASSES[arm], strict=True)
            if (mask & (groups == group)).any()
        ]
        total = sum(mass for _, mass in present)
        for mask, mass in present:
            result[mask] = mass / total / len(names) / int(mask.sum())
    if not np.isclose(result.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Global group/band weights are not normalized")
    return result
