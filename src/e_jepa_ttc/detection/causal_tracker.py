"""Causal multi-object association with explicit identity lifecycles.

The tracker consumes detector outputs only.  It never uses annotated identities,
future frames, or TTC targets.  Ambiguous assignments fail closed: every current
detection is retained, but receives a new generation that cannot inherit history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

Square = tuple[float, float, float, float]


class TrackState(StrEnum):
    """Lifecycle visible to downstream history owners."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True)
class DetectionObservation:
    """One causal detector observation in image coordinates."""

    detection_id: str
    class_name: str
    score: float
    event_square: Square
    rgb_square: Square | None = None
    roi_source: Literal["measured", "extrapolated"] = "measured"
    center_3d_m: tuple[float, float, float] | None = None
    size_lwh_m: tuple[float, float, float] | None = None
    yaw_rad: float | None = None
    velocity: tuple[float, ...] | None = None
    velocity_frame: str | None = None
    velocity_semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.detection_id or not self.class_name:
            raise ValueError("Detection and class identifiers must be nonempty")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be finite and in [0,1]")
        if self.roi_source not in ("measured", "extrapolated"):
            raise ValueError("ROI source must be measured or extrapolated")
        _validate_square(self.event_square)
        if self.rgb_square is not None:
            _validate_square(self.rgb_square)
        for name, values in (
            ("center_3d_m", self.center_3d_m),
            ("size_lwh_m", self.size_lwh_m),
        ):
            if values is not None and (
                len(values) != 3 or not all(math.isfinite(value) for value in values)
            ):
                raise ValueError(f"{name} must contain three finite values")
        if self.size_lwh_m is not None and any(value <= 0 for value in self.size_lwh_m):
            raise ValueError("size_lwh_m values must be positive")
        if self.yaw_rad is not None and not math.isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be finite")
        if self.velocity is not None and (
            not self.velocity or not all(math.isfinite(value) for value in self.velocity)
        ):
            raise ValueError("velocity must contain finite components")
        if self.velocity is not None and not self.velocity_frame:
            raise ValueError("Observed velocity requires an explicit coordinate frame")
        if self.velocity is not None and not self.velocity_semantics:
            raise ValueError("Observed velocity requires explicit semantics")


@dataclass(frozen=True)
class Track:
    """Current state of one generation; closed/lost tracks are never reattached."""

    track_id: str
    generation: int
    class_name: str
    state: TrackState
    hits: int
    missed: int
    first_timestamp_us: int
    last_timestamp_us: int
    event_square: Square
    rgb_square: Square | None

    @property
    def object_id(self) -> str:
        """Identity passed to PHASE caches and histories."""
        return f"{self.track_id}:g{self.generation}"


@dataclass(frozen=True)
class TrackAssignment:
    """Association result for a current detection."""

    detection: DetectionObservation
    track: Track
    association_status: Literal["new", "matched", "ambiguous"]
    association_cost: float | None
    ambiguity_reason: str

    @property
    def phase_admissible(self) -> bool:
        """Require a confirmed, unambiguous, measured observation."""
        return (
            self.track.state == TrackState.CONFIRMED
            and self.association_status != "ambiguous"
            and self.detection.roi_source == "measured"
        )


@dataclass(frozen=True)
class TrackerUpdate:
    """Assignments and all lifecycle transitions for one timestamp."""

    timestamp_us: int
    assignments: tuple[TrackAssignment, ...]
    tracks: tuple[Track, ...]


def _validate_square(square: Square) -> None:
    if len(square) != 4 or not all(math.isfinite(value) for value in square):
        raise ValueError("ROI must contain four finite coordinates")
    if square[2] <= square[0] or square[3] <= square[1]:
        raise ValueError("ROI must have positive width and height")


def box_iou(left: Square, right: Square) -> float:
    """Return image-plane IoU for validated xyxy boxes."""
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


class CausalMultiObjectTracker:
    """Globally associate detections while rejecting bidirectional ambiguity."""

    def __init__(
        self,
        *,
        minimum_iou: float = 0.3,
        ambiguity_margin: float = 0.05,
        confirm_hits: int = 2,
        lost_ttl_frames: int = 2,
        max_gap_us: int = 150_000,
    ) -> None:
        if not 0 <= minimum_iou <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("IoU and ambiguity margin must be in [0,1]")
        if confirm_hits < 1 or lost_ttl_frames < 0 or max_gap_us < 1:
            raise ValueError("Lifecycle thresholds must be positive")
        self.minimum_iou = minimum_iou
        self.ambiguity_margin = ambiguity_margin
        self.confirm_hits = confirm_hits
        self.lost_ttl_frames = lost_ttl_frames
        self.max_gap_us = max_gap_us
        self._last_timestamp_us: int | None = None
        self._tracks: dict[str, Track] = {}
        self._serial = 0
        self._generation = 0

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Return a stable snapshot including lost and closed generations."""
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    def _new_track(self, detection: DetectionObservation, timestamp_us: int) -> Track:
        self._serial += 1
        self._generation += 1
        state = TrackState.CONFIRMED if self.confirm_hits == 1 else TrackState.TENTATIVE
        track = Track(
            track_id=f"track_{self._serial:06d}",
            generation=self._generation,
            class_name=detection.class_name,
            state=state,
            hits=1,
            missed=0,
            first_timestamp_us=timestamp_us,
            last_timestamp_us=timestamp_us,
            event_square=detection.event_square,
            rgb_square=detection.rgb_square,
        )
        self._tracks[track.object_id] = track
        return track

    def update(self, timestamp_us: int, detections: list[DetectionObservation]) -> TrackerUpdate:
        """Consume exactly one strictly newer frame and return current assignments."""
        if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int):
            raise ValueError("timestamp_us must be integer microseconds")
        if self._last_timestamp_us is not None and timestamp_us <= self._last_timestamp_us:
            raise ValueError("Tracker timestamps must increase strictly")
        ids = [item.detection_id for item in detections]
        if len(ids) != len(set(ids)):
            raise ValueError("Detection IDs must be unique within a frame")

        if (
            self._last_timestamp_us is not None
            and timestamp_us - self._last_timestamp_us > self.max_gap_us
        ):
            self._tracks = {
                key: replace(track, state=TrackState.CLOSED)
                if track.state != TrackState.CLOSED
                else track
                for key, track in self._tracks.items()
            }

        candidates = [
            track
            for track in self.tracks
            if track.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]
        cost = np.full((len(candidates), len(detections)), np.inf, dtype=np.float64)
        for ti, track in enumerate(candidates):
            for di, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                overlap = box_iou(track.event_square, detection.event_square)
                if overlap >= self.minimum_iou:
                    cost[ti, di] = 1.0 - overlap

        ambiguous_reasons: dict[int, set[str]] = {}
        for di in range(len(detections)):
            column = cost[:, di]
            finite = column[np.isfinite(column)]
            if (
                len(finite) >= 2
                and np.partition(finite, 1)[1] - finite.min() < self.ambiguity_margin
            ):
                ambiguous_reasons.setdefault(di, set()).add("detection_to_tracks_margin")
        for ti in range(len(candidates)):
            row = cost[ti, :]
            finite_indices = np.flatnonzero(np.isfinite(row))
            if len(finite_indices) < 2:
                continue
            best = float(row[finite_indices].min())
            if np.partition(row[finite_indices], 1)[1] - best < self.ambiguity_margin:
                for di in finite_indices:
                    if float(row[di]) - best < self.ambiguity_margin:
                        ambiguous_reasons.setdefault(int(di), set()).add(
                            "track_to_detections_margin"
                        )

        proposed: dict[int, tuple[int, float]] = {}
        if cost.size and np.isfinite(cost).any():
            safe = np.where(np.isfinite(cost), cost, 1e6)
            rows, columns = linear_sum_assignment(safe)
            for ti, di in zip(rows.tolist(), columns.tolist(), strict=True):
                if np.isfinite(cost[ti, di]):
                    proposed[di] = (ti, float(cost[ti, di]))

        accepted: dict[int, tuple[int, float]] = {}
        ambiguous = {
            index: "+".join(sorted(reasons)) for index, reasons in ambiguous_reasons.items()
        }
        for di, (ti, value) in proposed.items():
            if di not in ambiguous:
                accepted[di] = (ti, value)

        matched_keys: set[str] = set()
        assignments = []
        for di, detection in enumerate(detections):
            if di in accepted:
                ti, value = accepted[di]
                old = candidates[ti]
                hits = old.hits + 1
                state = TrackState.CONFIRMED if hits >= self.confirm_hits else TrackState.TENTATIVE
                track = replace(
                    old,
                    state=state,
                    hits=hits,
                    missed=0,
                    last_timestamp_us=timestamp_us,
                    event_square=detection.event_square,
                    rgb_square=detection.rgb_square,
                )
                self._tracks[track.object_id] = track
                matched_keys.add(track.object_id)
                assignments.append(TrackAssignment(detection, track, "matched", value, ""))
            else:
                track = self._new_track(detection, timestamp_us)
                status = "ambiguous" if di in ambiguous else "new"
                assignments.append(
                    TrackAssignment(detection, track, status, None, ambiguous.get(di, ""))
                )
                matched_keys.add(track.object_id)

        for key, track in tuple(self._tracks.items()):
            if key in matched_keys or track.state == TrackState.CLOSED:
                continue
            missed = track.missed + 1
            state = TrackState.CLOSED if missed > self.lost_ttl_frames else TrackState.LOST
            self._tracks[key] = replace(track, state=state, missed=missed)

        self._last_timestamp_us = timestamp_us
        return TrackerUpdate(timestamp_us, tuple(assignments), self.tracks)
