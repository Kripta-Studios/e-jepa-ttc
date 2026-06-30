"""Typed data contracts used across the event TTC pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventBatch:
    """A temporally contiguous event batch.

    Arrays are one-dimensional and aligned by index. Polarity is normalized to
    `{-1, +1}` by validation/construction helpers.
    """

    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    width: int
    height: int
    sequence_id: str
    t_start_us: int
    t_end_us: int

    @property
    def num_events(self) -> int:
        """Return the number of events in the batch."""

        return int(self.x.shape[0])

    @property
    def duration_us(self) -> int:
        """Return the batch duration in microseconds."""

        return int(self.t_end_us - self.t_start_us)

    @classmethod
    def empty(
        cls,
        *,
        width: int,
        height: int,
        sequence_id: str,
        t_start_us: int,
        t_end_us: int,
    ) -> EventBatch:
        """Create an explicitly empty batch."""

        return cls(
            x=np.empty(0, dtype=np.int32),
            y=np.empty(0, dtype=np.int32),
            t_us=np.empty(0, dtype=np.int64),
            polarity=np.empty(0, dtype=np.int8),
            width=width,
            height=height,
            sequence_id=sequence_id,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
        )


@dataclass(frozen=True)
class TTCWindowSample:
    """One context window and optional future/target supervision."""

    context_events: EventBatch
    future_events: dict[int, EventBatch]
    ttc_seconds: float | None
    collision_within: dict[float, bool] | None
    object_bbox: np.ndarray | None
    object_mask: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetSequence:
    """Manifest entry for one event sequence."""

    dataset: str
    sequence_id: str
    local_path: str
    event_hdf5: str
    gt_hdf5: str | None = None
    ttc_csv: str | None = None
    label_dir: str | None = None
    scenario_family: str | None = None
    speed_bucket: str | None = None
    target_type: str | None = None
    split_group: str | None = None
    size_bytes: int | None = None
    source_url: str | None = None
    retrieval_date: str | None = None
    license: str | None = None
    citation: str | None = None
    original_filename: str | None = None
    sha256: str | None = None
    remote_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a manifest-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSequence:
        """Build from a manifest dictionary."""

        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        known = {key: value for key, value in data.items() if key in allowed}
        unknown = {key: value for key, value in data.items() if key not in allowed}
        extra = dict(known.get("extra") or {})
        extra.update(unknown)
        known["extra"] = extra
        return cls(**known)

    def resolve(self, field_name: str) -> Path | None:
        """Resolve a path field relative to the sequence local path."""

        value = getattr(self, field_name)
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(self.local_path) / path


@dataclass(frozen=True)
class TemporalIndexEntry:
    """One indexed window anchored at a reference timestamp."""

    sequence_id: str
    timestamp_us: int
    context_start_us: int
    context_end_us: int
    horizons_us: dict[int, tuple[int, int]]
    ttc_seconds: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON/YAML-safe values."""

        data = asdict(self)
        data["horizons_us"] = {
            str(horizon): [int(start), int(end)]
            for horizon, (start, end) in self.horizons_us.items()
        }
        return data
