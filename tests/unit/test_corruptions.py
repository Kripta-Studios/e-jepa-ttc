from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.corruptions import (
    EventCorruptionSpec,
    corrupt_event_batch,
)


def _events() -> EventBatch:
    return EventBatch(
        x=np.arange(20, dtype=np.int32) % 8,
        y=np.arange(20, dtype=np.int32) % 6,
        t_us=np.arange(20, dtype=np.int64) * 50 + 1_000,
        polarity=np.where(np.arange(20) % 2, 1, -1).astype(np.int8),
        width=8,
        height=6,
        sequence_id="fixture",
        t_start_us=1_000,
        t_end_us=2_000,
    )


@pytest.mark.parametrize(
    ("kind", "severity"),
    [
        ("event_dropout", 0.3),
        ("timestamp_jitter_us", 100.0),
        ("background_event_rate", 0.5),
        ("hot_pixel_fraction", 0.1),
        ("dead_pixel_fraction", 0.2),
        ("polarity_drop_positive", 1.0),
        ("polarity_drop_negative", 1.0),
        ("temporal_window_fraction", 0.5),
        ("spatial_crop_fraction", 0.75),
    ],
)
def test_corruptions_are_deterministic_valid_and_sorted(kind: str, severity: float) -> None:
    spec = EventCorruptionSpec(kind=kind, severity=severity, seed=7)
    first = corrupt_event_batch(_events(), spec)
    second = corrupt_event_batch(_events(), spec)

    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.t_us, second.t_us)
    assert np.all(np.diff(first.t_us) >= 0)
    assert np.all((first.x >= 0) & (first.x < first.width))
    assert np.all((first.y >= 0) & (first.y < first.height))
    assert set(np.unique(first.polarity)).issubset({-1, 1})
    assert np.all((first.t_us >= first.t_start_us) & (first.t_us < first.t_end_us))


def test_dropout_and_polarity_corruption_have_expected_effect() -> None:
    dropped = corrupt_event_batch(
        _events(),
        EventCorruptionSpec(kind="event_dropout", severity=1.0),
    )
    positive_removed = corrupt_event_batch(
        _events(),
        EventCorruptionSpec(kind="polarity_drop_positive", severity=1.0),
    )

    assert dropped.num_events == 0
    assert np.all(positive_removed.polarity == -1)

