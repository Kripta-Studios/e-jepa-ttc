"""TRAIN-head-only mass ablations with unchanged query population and sampler.

M0 preserves the exact native cached mass. M1 gives each group's present TTC
bands the historical 0.5/0.3/0.1/0.1 relative masses. M2 redistributes *within*
crucial/long bands, retaining their total mass and bounding per-query changes.
These are training distributions, NOT replacement evaluation aggregations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassPlans:
    native: np.ndarray
    benchmark: np.ndarray
    subrange: np.ndarray


def _bounded_redistribute(base: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Project positive desired weights, preserving mass and ratios in [1/4,4]."""
    total = float(base.sum())
    lower, upper = 0.25 * base, 4.0 * base
    lo, hi = 0.0, float(np.max(upper / desired)) * 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        value = np.clip(mid * desired, lower, upper)
        if float(value.sum()) < total:
            lo = mid
        else:
            hi = mid
    value = np.clip(((lo + hi) / 2.0) * desired, lower, upper)
    if not np.isclose(value.sum(), total, rtol=1e-12, atol=1e-15):
        raise ArithmeticError("Bounded mass projection did not conserve mass")
    return value


def fit_mass_plans(
    target_ttc: np.ndarray, groups: np.ndarray, native_mass: np.ndarray, *, fit_role: str
) -> MassPlans:
    """Create a frozen H-role mass plan. V/Dev32/FCWD labels are not admissible.

    All targets must lie in the original signed training bands. Out-of-support
    data require a new declared protocol; they are never silently dropped.
    Caller must also verify the H-role cache and producer lineage manifests.
    """
    if fit_role != "H":
        raise ValueError("Only the verified head-training role H may fit mass plans")
    target = np.asarray(target_ttc, dtype=np.float64)
    group = np.asarray(groups).astype(str)
    raw_native = np.asarray(native_mass)
    native = np.asarray(native_mass, dtype=np.float64)
    if (
        target.ndim != 1
        or not len(target)
        or group.shape != target.shape
        or native.shape != target.shape
    ):
        raise ValueError("Nonempty aligned one-dimensional arrays required")
    if not np.isfinite(target).all() or not np.isfinite(native).all() or np.any(native <= 0):
        raise ValueError("Finite targets and strictly positive masses required")
    if not np.isclose(native.sum(), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Native full-population mass must sum to one")
    masks = (
        (target > 0.1) & (target <= 3),
        (target > 3) & (target <= 6),
        (target > 6) & (target <= 10),
        (target > -10) & (target < 0),
    )
    if not np.logical_or.reduce(masks).all():
        raise ValueError("Targets outside signed benchmark TRAIN support; no row deletion")
    benchmark = np.zeros_like(native)
    weights = (0.5, 0.3, 0.1, 0.1)
    for name in np.unique(group):
        in_group = group == name
        group_mass = float(native[in_group].sum())
        present = [
            (in_group & mask, weight)
            for mask, weight in zip(masks, weights, strict=True)
            if (in_group & mask).any()
        ]
        denom = sum(weight for _, weight in present)
        for mask, weight in present:
            benchmark[mask] = group_mass * (weight / denom) / int(mask.sum())
    refined = benchmark.copy()
    for name in np.unique(group):
        in_group = group == name
        for band_mask, intervals in (
            (masks[0], ((0.1, 1), (1, 2), (2, 3))),
            (masks[2], ((6, 8), (8, 10))),
        ):
            selected = in_group & band_mask
            ids = np.flatnonzero(selected)
            if not len(ids):
                continue
            submasks = [(target[ids] > low) & (target[ids] <= high) for low, high in intervals]
            submasks = [m for m in submasks if m.any()]
            desired = np.zeros(len(ids), dtype=np.float64)
            total = float(benchmark[ids].sum())
            for mask in submasks:
                desired[mask] = total / (len(submasks) * int(mask.sum()))
            refined[ids] = _bounded_redistribute(benchmark[ids], desired)
    # M0 is copied without re-normalization/rounding: baseline identity matters.
    return MassPlans(raw_native.copy(), benchmark, refined)
