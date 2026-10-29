"""Event visualization primitives that preserve the declared resolution."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.event_count import encode_event_count


def render_event_count(events: EventBatch) -> np.ndarray:
    """Return a positive/negative count image for plotting or artifact storage."""

    return encode_event_count(events, log1p=False)


__all__ = ["render_event_count"]
