"""Public augmentation entry points for event-only experiments."""

from e_jepa_ttc.representations.corruptions import (
    CORRUPTION_KINDS,
    EventCorruptionSpec,
    corrupt_event_batch,
)

__all__ = ["CORRUPTION_KINDS", "EventCorruptionSpec", "corrupt_event_batch"]
