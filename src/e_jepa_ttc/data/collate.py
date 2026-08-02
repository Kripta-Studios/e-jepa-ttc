"""Variable-length event collation that keeps raw events as lists."""

from __future__ import annotations

from typing import Any

import numpy as np

from e_jepa_ttc.data.types import TTCWindowSample


def collate_ttc_windows(samples: list[TTCWindowSample]) -> dict[str, Any]:
    """Collate windows without padding raw events or silently dropping empties."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    return {
        "context_events": [sample.context_events for sample in samples],
        "future_events": [sample.future_events for sample in samples],
        "ttc_seconds": np.asarray(
            [np.nan if sample.ttc_seconds is None else sample.ttc_seconds for sample in samples],
            dtype=np.float32,
        ),
        "collision_within": [sample.collision_within for sample in samples],
        "object_bbox": [sample.object_bbox for sample in samples],
        "object_mask": [sample.object_mask for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }


__all__ = ["collate_ttc_windows"]
