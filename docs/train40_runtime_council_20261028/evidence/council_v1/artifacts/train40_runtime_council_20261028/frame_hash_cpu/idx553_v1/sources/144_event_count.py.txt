"""Event count representation."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch


def encode_event_count(
    events: EventBatch,
    *,
    log1p: bool = True,
    duration_normalize: bool = False,
) -> np.ndarray:
    """Encode events as positive and negative count images `[2,H,W]`."""

    image = np.zeros((2, events.height, events.width), dtype=np.float32)
    if events.num_events == 0:
        return image

    pos = events.polarity > 0
    np.add.at(image[0], (events.y[pos], events.x[pos]), 1.0)
    np.add.at(image[1], (events.y[~pos], events.x[~pos]), 1.0)

    if duration_normalize and events.duration_us > 0:
        image *= 1_000_000.0 / float(events.duration_us)
    if log1p:
        image = np.log1p(image)
    return image.astype(np.float32)
