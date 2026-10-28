"""Time surface representation."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch


def encode_time_surface(events: EventBatch, *, tau_ms: float = 30.0) -> np.ndarray:
    """Encode last-event recency as `exp(-(t_end - t_last) / tau)`."""

    if tau_ms <= 0:
        msg = "tau_ms must be positive."
        raise ValueError(msg)

    last_t = np.full((2, events.height, events.width), -np.inf, dtype=np.float64)
    if events.num_events > 0:
        channel = np.where(events.polarity > 0, 0, 1)
        for c, y, x, t in zip(channel, events.y, events.x, events.t_us, strict=True):
            last_t[int(c), int(y), int(x)] = max(last_t[int(c), int(y), int(x)], float(t))

    age_us = float(events.t_end_us) - last_t
    surface = np.exp(-age_us / (tau_ms * 1000.0))
    surface[~np.isfinite(surface)] = 0.0
    return surface.astype(np.float32)
