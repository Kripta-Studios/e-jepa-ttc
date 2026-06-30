"""Representation protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from e_jepa_ttc.data.types import EventBatch


class EventRepresentation(Protocol):
    """Protocol for event encoders."""

    def encode(self, events: EventBatch) -> np.ndarray:
        """Encode an event batch to an array."""
