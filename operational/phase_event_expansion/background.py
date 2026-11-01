"""Independent static-contour contamination, with unchanged foreground motion."""

from dataclasses import replace

import numpy as np

from operational.phase_event_expansion.synthetic import SyntheticFixture, make_fixture


def background_fixture(seed: int, fraction: float) -> tuple[SyntheticFixture, np.ndarray]:
    """Return sorted mixture and an audit-only mask never supplied to the estimator."""
    if not 0 <= fraction < 1:
        raise ValueError("Background fraction must be finite in [0,1)")
    original = make_fixture(seed, "expansion")
    count = round(len(original.events) * fraction / (1 - fraction))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 41031]))
    side = rng.integers(0, 4, count)
    along = rng.uniform(-1, 1, count)
    points = np.column_stack((0.44 * along, 0.40 * along))
    points[side == 0, 0] = -0.44
    points[side == 1, 0] = 0.44
    points[side == 2, 1] = -0.40
    points[side == 3, 1] = 0.40
    points += rng.normal(0, 0.0005, points.shape)
    clock = rng.integers(original.query_us - 40_000, original.query_us + 1, count)
    background = np.column_stack((128 + 256 * points, clock, np.where(side % 2, 1, -1)))
    events = np.concatenate((original.events, background))
    foreground = np.arange(len(events)) < len(original.events)
    order = np.argsort(events[:, 2], kind="stable")
    return replace(original, events=events[order]), foreground[order]
