"""Deterministic raw-event corruptions for robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from e_jepa_ttc.data.types import EventBatch

CORRUPTION_KINDS = (
    "none",
    "event_dropout",
    "timestamp_jitter_us",
    "background_event_rate",
    "hot_pixel_fraction",
    "dead_pixel_fraction",
    "polarity_drop_positive",
    "polarity_drop_negative",
    "temporal_window_fraction",
    "spatial_crop_fraction",
)


@dataclass(frozen=True)
class EventCorruptionSpec:
    """One reproducible corruption and its scalar severity."""

    kind: str = "none"
    severity: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in CORRUPTION_KINDS:
            msg = f"Unsupported event corruption {self.kind!r}."
            raise ValueError(msg)
        if not np.isfinite(self.severity) or self.severity < 0:
            msg = "Corruption severity must be finite and non-negative."
            raise ValueError(msg)
        unit_interval = {
            "event_dropout",
            "hot_pixel_fraction",
            "dead_pixel_fraction",
            "temporal_window_fraction",
            "spatial_crop_fraction",
        }
        if self.kind in unit_interval and self.severity > 1:
            msg = f"{self.kind} severity must not exceed one."
            raise ValueError(msg)
        positive_fraction = {"temporal_window_fraction", "spatial_crop_fraction"}
        if self.kind in positive_fraction and self.severity <= 0:
            msg = f"{self.kind} severity must be in (0, 1]."
            raise ValueError(msg)


def corrupt_event_batch(
    events: EventBatch,
    spec: EventCorruptionSpec,
    *,
    seed_offset: int = 0,
) -> EventBatch:
    """Apply a corruption while preserving a valid, time-sorted EventBatch."""

    polarity_kinds = {"polarity_drop_positive", "polarity_drop_negative"}
    if (
        spec.kind == "none"
        or events.num_events == 0
        or (spec.severity == 0 and spec.kind not in polarity_kinds)
    ):
        return events
    rng = np.random.default_rng(np.uint64(spec.seed) + np.uint64(seed_offset))
    x = events.x.copy()
    y = events.y.copy()
    timestamps = events.t_us.copy()
    polarity = events.polarity.copy()
    start_us = events.t_start_us
    end_us = events.t_end_us

    if spec.kind == "event_dropout":
        keep = rng.random(events.num_events) >= spec.severity
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]
    elif spec.kind == "timestamp_jitter_us":
        jitter = rng.normal(0.0, spec.severity, events.num_events).round().astype(np.int64)
        timestamps = np.clip(timestamps + jitter, start_us, max(start_us, end_us - 1))
    elif spec.kind == "background_event_rate":
        count = int(round(events.num_events * spec.severity))
        if count > 0:
            x = np.concatenate((x, rng.integers(0, events.width, count, dtype=np.int32)))
            y = np.concatenate((y, rng.integers(0, events.height, count, dtype=np.int32)))
            timestamps = np.concatenate(
                (timestamps, rng.integers(start_us, end_us, count, dtype=np.int64))
            )
            polarity = np.concatenate(
                (polarity, rng.choice(np.asarray([-1, 1], dtype=np.int8), count))
            )
    elif spec.kind == "hot_pixel_fraction":
        pixel_count = max(1, int(round(events.width * events.height * spec.severity)))
        selected = rng.choice(events.width * events.height, pixel_count, replace=False)
        count = max(pixel_count, int(round(events.num_events * spec.severity)))
        hot = rng.choice(selected, count, replace=True)
        x = np.concatenate((x, (hot % events.width).astype(np.int32)))
        y = np.concatenate((y, (hot // events.width).astype(np.int32)))
        timestamps = np.concatenate(
            (timestamps, rng.integers(start_us, end_us, count, dtype=np.int64))
        )
        polarity = np.concatenate((polarity, rng.choice(np.asarray([-1, 1], dtype=np.int8), count)))
    elif spec.kind == "dead_pixel_fraction":
        count = int(round(events.width * events.height * spec.severity))
        dead = rng.choice(events.width * events.height, count, replace=False)
        linear = y.astype(np.int64) * events.width + x
        keep = ~np.isin(linear, dead)
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]
    elif spec.kind == "polarity_drop_positive":
        keep = polarity < 0
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]
    elif spec.kind == "polarity_drop_negative":
        keep = polarity > 0
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]
    elif spec.kind == "temporal_window_fraction":
        duration = max(1, int(round(events.duration_us * spec.severity)))
        start_us = end_us - duration
        keep = timestamps >= start_us
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]
    elif spec.kind == "spatial_crop_fraction":
        crop_width = max(1, int(round(events.width * spec.severity)))
        crop_height = max(1, int(round(events.height * spec.severity)))
        x_min = (events.width - crop_width) // 2
        y_min = (events.height - crop_height) // 2
        keep = (x >= x_min) & (x < x_min + crop_width) & (y >= y_min) & (y < y_min + crop_height)
        x, y, timestamps, polarity = x[keep], y[keep], timestamps[keep], polarity[keep]

    if timestamps.size:
        order = np.argsort(timestamps, kind="stable")
        x, y, timestamps, polarity = x[order], y[order], timestamps[order], polarity[order]
    return EventBatch(
        x=x.astype(np.int32, copy=False),
        y=y.astype(np.int32, copy=False),
        t_us=timestamps.astype(np.int64, copy=False),
        polarity=polarity.astype(np.int8, copy=False),
        width=events.width,
        height=events.height,
        sequence_id=events.sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )


__all__ = ["CORRUPTION_KINDS", "EventCorruptionSpec", "corrupt_event_batch"]
