"""Lifecycle-safe adapter from predicted detections to MultiObjectTrain40Service.

This module is an integration boundary.  It reports detector outputs even when
tracking or TTC abstains and never interprets phase quantiles as TTC intervals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Protocol

import numpy as np

from e_jepa_ttc.detection.causal_tracker import (
    CausalMultiObjectTracker,
    DetectionObservation,
    TrackAssignment,
)
from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory


class Train40MultiObjectRuntime(Protocol):
    """Structural subset of the actual MultiObjectTrain40Service API."""

    max_objects: int
    history: Any

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        objects: Sequence[ObjectRegion],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class JointObjectPrediction:
    """One retained detection with optional signed TTC and native phase bounds."""

    detection_id: str
    class_name: str
    detection_score: float
    track_id: str
    generation: int
    track_state: str
    association_status: str
    roi_source: str
    event_square: tuple[float, float, float, float]
    rgb_square: tuple[float, float, float, float] | None
    center_3d_m: tuple[float, float, float] | None
    size_lwh_m: tuple[float, float, float] | None
    yaw_rad: float | None
    velocity: tuple[float, ...] | None
    velocity_frame: str | None
    velocity_semantics: str | None
    ttc_status: str
    ttc_reason: str
    ttc_signed_s: float | None
    point_phase: float | None
    phase_q10: float | None
    phase_q90: float | None
    phase_bounds_semantics: str
    runtime_output: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record without changing signed values."""
        return asdict(self)


def _finite_optional(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Runtime {field} must be numeric when present")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Runtime {field} must be finite when present")
    return result


def _native_fields(
    output: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    predictions = output.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("TRAIN40 object output must contain a predictions mapping")
    preferred = predictions.get("PHASE_ONLY_TRAIN40_S7")
    if preferred is None:
        return None, None, None, None
    ttc = _finite_optional(preferred, "TTC")
    assert ttc is not None
    phase = output.get("phase")
    phase_map = phase if isinstance(phase, Mapping) else output
    point = _finite_optional(phase_map.get("point_phase"), "point phase")
    q10 = _finite_optional(phase_map.get("q10"), "phase q10")
    q90 = _finite_optional(phase_map.get("q90"), "phase q90")
    return ttc, point, q10, q90


class DetectionPhaseService:
    """Own causal association and invoke TRAIN40 exactly once per raw packet."""

    def __init__(
        self,
        runtime: Train40MultiObjectRuntime,
        *,
        tracker: CausalMultiObjectTracker | None = None,
        minimum_track_history_us: int = 600_000,
    ) -> None:
        if not isinstance(runtime.max_objects, int) or runtime.max_objects < 1:
            raise ValueError("Runtime max_objects must be a positive integer")
        self.runtime = runtime
        self.tracker = tracker or CausalMultiObjectTracker()
        if minimum_track_history_us < 0:
            raise ValueError("minimum_track_history_us must be nonnegative")
        self.minimum_track_history_us = minimum_track_history_us
        self._last_end_us: int | None = None
        self._closed = False
        self._failed = False
        self._runtime_closed = False

    def close(self) -> None:
        """Close the injected service once."""
        if not self._runtime_closed:
            self.runtime.close()
            self._runtime_closed = True
        self._closed = True

    def _abstention(self, assignment: TrackAssignment, reason: str) -> JointObjectPrediction:
        item, track = assignment.detection, assignment.track
        return JointObjectPrediction(
            item.detection_id,
            item.class_name,
            item.score,
            track.track_id,
            track.generation,
            str(track.state),
            assignment.association_status,
            item.roi_source,
            item.event_square,
            item.rgb_square,
            item.center_3d_m,
            item.size_lwh_m,
            item.yaw_rad,
            item.velocity,
            item.velocity_frame,
            item.velocity_semantics,
            "unavailable",
            reason,
            None,
            None,
            None,
            None,
            "native_phase_bounds_absent",
            None,
        )

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        detections: Sequence[DetectionObservation],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]:
        """Consume one packet and return every detection, including abstentions."""
        if self._closed:
            raise RuntimeError("Detection/PHASE service is closed")
        if self._failed:
            raise RuntimeError("Detection/PHASE service failed after packet ownership transfer")
        if isinstance(start_us, bool) or isinstance(end_us, bool) or end_us <= start_us:
            raise ValueError("Packet timestamps must be increasing integer microseconds")
        if not isinstance(start_us, int) or not isinstance(end_us, int):
            raise ValueError("Packet timestamps must be increasing integer microseconds")
        if self._last_end_us is not None and start_us != self._last_end_us:
            raise ValueError("Packets must be contiguous and consumed exactly once")

        tracked = self.tracker.update(end_us, list(detections))
        selected = [
            item
            for item in tracked.assignments
            if item.phase_admissible
            and end_us - item.track.first_timestamp_us >= self.minimum_track_history_us
        ]
        selected.sort(key=lambda item: (-item.detection.score, item.detection.detection_id))
        admitted = selected[: self.runtime.max_objects]
        admitted_ids = {item.detection.detection_id for item in admitted}
        regions = [
            ObjectRegion(
                item.track.object_id,
                item.detection.event_square,
                item.detection.rgb_square,
            )
            for item in admitted
        ]

        try:
            if regions:
                runtime_receipt = self.runtime.update(
                    raw_packet, start_us, end_us, regions, rgb_history
                )
                outputs = runtime_receipt.get("objects")
                if not isinstance(outputs, list) or len(outputs) != len(regions):
                    raise ValueError("TRAIN40 result count differs from admitted regions")
                by_id = {}
                for output in outputs:
                    if not isinstance(output, dict) or output.get("object_id") in by_id:
                        raise ValueError("TRAIN40 outputs need unique object_id values")
                    by_id[output.get("object_id")] = output
                if set(by_id) != {item.track.object_id for item in admitted}:
                    raise ValueError("TRAIN40 object identities differ from tracker generations")
            else:
                if bool(getattr(self.runtime, "supports_empty_updates", False)):
                    runtime_receipt = self.runtime.update(
                        raw_packet, start_us, end_us, (), rgb_history
                    )
                else:
                    # Historical MultiObjectTrain40Service rejects zero objects.
                    # Its history still needs every packet for a future track.
                    self.runtime.history.ingest(raw_packet, start_us, end_us)
                    runtime_receipt = {
                        "objects": [],
                        "object_count": 0,
                        "targets_read": False,
                        "optimizer_updates": 0,
                        "zero_object_packet_ingested": True,
                    }
                by_id = {}
        except BaseException:
            # Packet ownership may already have transferred to EventHistory.
            # Prohibit a retry that could ingest the same interval twice.
            self._failed = True
            raise
        self._last_end_us = end_us

        joint = []
        try:
            for assignment in tracked.assignments:
                detection = assignment.detection
                if detection.detection_id not in admitted_ids:
                    if assignment.association_status == "ambiguous":
                        reason = f"identity_ambiguous:{assignment.ambiguity_reason}"
                    elif detection.roi_source == "extrapolated":
                        reason = "roi_extrapolated"
                    elif (
                        assignment.phase_admissible
                        and end_us - assignment.track.first_timestamp_us
                        < self.minimum_track_history_us
                    ):
                        reason = "insufficient_track_history"
                    elif not assignment.phase_admissible:
                        reason = "track_not_confirmed"
                    else:
                        reason = "runtime_object_capacity"
                    joint.append(self._abstention(assignment, reason))
                    continue
                output = by_id[assignment.track.object_id]
                ttc, point, q10, q90 = _native_fields(output)
                if ttc is None:
                    joint.append(
                        JointObjectPrediction(
                            detection.detection_id,
                            detection.class_name,
                            detection.score,
                            assignment.track.track_id,
                            assignment.track.generation,
                            str(assignment.track.state),
                            assignment.association_status,
                            detection.roi_source,
                            detection.event_square,
                            detection.rgb_square,
                            detection.center_3d_m,
                            detection.size_lwh_m,
                            detection.yaw_rad,
                            detection.velocity,
                            detection.velocity_frame,
                            detection.velocity_semantics,
                            "unavailable",
                            "runtime_missing_declared_final_ttc",
                            None,
                            None,
                            None,
                            None,
                            "native_phase_bounds_absent",
                            dict(output),
                        )
                    )
                    continue
                joint.append(
                    JointObjectPrediction(
                        detection.detection_id,
                        detection.class_name,
                        detection.score,
                        assignment.track.track_id,
                        assignment.track.generation,
                        str(assignment.track.state),
                        assignment.association_status,
                        detection.roi_source,
                        detection.event_square,
                        detection.rgb_square,
                        detection.center_3d_m,
                        detection.size_lwh_m,
                        detection.yaw_rad,
                        detection.velocity,
                        detection.velocity_frame,
                        detection.velocity_semantics,
                        "available",
                        "",
                        ttc,
                        point,
                        q10,
                        q90,
                        (
                            "native_phase_quantile_bounds_not_continuous_ttc_interval"
                            if q10 is not None and q90 is not None
                            else "native_phase_bounds_absent"
                        ),
                        dict(output),
                    )
                )
        except BaseException:
            self._failed = True
            raise
        return {
            "schema": "detection_phase_multiobject_v1",
            "timestamp_us": end_us,
            "detections": [item.to_dict() for item in joint],
            "tracks": [asdict(item) for item in tracked.tracks],
            "runtime": runtime_receipt,
            "packet_ingested_count": 1,
            "targets_read": False,
            "optimizer_updates": 0,
            "generalization_claim": False,
        }
