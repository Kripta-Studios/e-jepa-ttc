"""Sparse event token representation."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch


def encode_sparse_tokens(
    events: EventBatch,
    *,
    max_tokens: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Encode events as `[N, 6]` tokens.

    Columns: x_norm, y_norm, t_norm, polarity, local_density_placeholder,
    inter_event_time_norm.
    """

    if max_tokens <= 0:
        msg = "max_tokens must be positive."
        raise ValueError(msg)
    if events.num_events == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if events.num_events > max_tokens:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(events.num_events, size=max_tokens, replace=False))
    else:
        indices = np.arange(events.num_events)

    x = events.x[indices].astype(np.float32) / max(float(events.width - 1), 1.0)
    y = events.y[indices].astype(np.float32) / max(float(events.height - 1), 1.0)
    t = (events.t_us[indices].astype(np.float64) - float(events.t_start_us)) / max(
        float(events.duration_us), 1.0
    )
    p = events.polarity[indices].astype(np.float32)
    inter = np.diff(events.t_us[indices], prepend=events.t_us[indices][0]).astype(np.float32)
    inter = inter / max(float(events.duration_us), 1.0)
    density = np.full_like(
        x, fill_value=float(events.num_events) / max(events.width * events.height, 1)
    )
    tokens = np.column_stack([x, y, t.astype(np.float32), p, density, inter])
    return tokens.astype(np.float32)
