"""Explicit synthetic contour-event fixtures, not a photometric event simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticFixture:
    """Events and separate ground truth; only events/ROI/clock enter the solver."""

    events: np.ndarray
    roi: np.ndarray
    query_us: int
    q: float
    translation: np.ndarray
    kind: str
    seed: int


def make_fixture(
    seed: int, kind: str, *, count: int = 4096, jitter: float = 0.0005
) -> SyntheticFixture:
    """Generate a 40ms window with a single current 256px ROI.

    Samples follow constant-velocity 3D translation projected into the image.
    This idealized contour process is deliberately not a realistic sensor claim.
    """
    if kind not in ("expansion", "receding", "translation", "stationary", "noise", "edge"):
        raise ValueError("Unknown synthetic fixture")
    rng = np.random.default_rng(seed)
    q = float(rng.uniform(0.75, 5.0))
    if kind == "receding":
        q = -q
    if kind in ("translation", "stationary", "noise", "edge"):
        q = 0.0
    translation = rng.uniform(-0.6, 0.6, 2)
    if kind in ("stationary", "noise", "edge"):
        translation[:] = 0
    query_us = 100_000
    clock = np.sort(rng.integers(query_us - 40_000, query_us + 1, count))
    age = (query_us - clock) / 1e6
    side = rng.integers(0, 4, count)
    along = rng.uniform(-1, 1, count)
    endpoint = np.column_stack((0.35 * along, 0.22 * along))
    endpoint[side == 0, 0] = -0.35
    endpoint[side == 1, 0] = 0.35
    endpoint[side == 2, 1] = -0.22
    endpoint[side == 3, 1] = 0.22
    points = (endpoint + translation * age[:, None]) / (1 + q * age[:, None])
    if kind == "noise":
        points = rng.uniform(-0.45, 0.45, (count, 2))
    elif kind == "edge":
        points[:, 0] = np.linspace(-0.35, 0.35, count)
        points[:, 1] = 0
    points += rng.normal(0, jitter, points.shape)
    polarity = np.where(side % 2, 1, -1)
    events = np.column_stack((128 + 256 * points, clock, polarity))
    return SyntheticFixture(
        events, np.array([0.0, 0.0, 256.0, 256.0]), query_us, q, translation, kind, seed
    )
