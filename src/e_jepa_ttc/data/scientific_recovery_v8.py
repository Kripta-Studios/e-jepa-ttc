"""Isolated causal temporal frontends for Scientific Recovery V8.

The Garl time-volume wrapper deliberately reuses the checked-in official
preprocessing helper, but it is only an isolated time-volume frontend and not
full Garl-TTC preprocessing/model parity.

EXP6 freezes the filter-state equation subset of EV-TTC at
``59c498b71ae526bc2d7e570c82a078306a996b93``.  In the source implementation,
an event in 0.2 ms bin ``j`` contributes signed spatial mass times
``alpha * (1 - alpha)**(-j)`` to the current 7 ms window; at the output
snapshot the complete state is multiplied by ``(1 - alpha)**35``.  A
boundary-triggering event is snapshotted *before* it is inserted at bin zero
of the next window.  The local raster contract uses integer event coordinates,
already-normalized polarity ``{-1, +1}``, and a stable ``EventBatch.t_start_us``
origin.  EV-TTC's ROS callback triggers an output from its first event after
an elapsed ``>= 7 ms`` condition, so a real event can overshoot the nominal
boundary.  V8 intentionally uses fixed-grid endpoints instead.  It therefore
proves the filter-state equation only under those frozen raster assumptions—not
exact official trigger timing, full ROS scheduling, lens correction, half
precision, bilinear placement, crop, or downsampling parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_timevolume_roi_np,
)
from e_jepa_ttc.data.types import EventBatch

GARL_TIMEVOL_SOURCE = "isolated_garl_timevolume_frontend_not_full_garl_parity"
EXP6_SOURCE = "evttc_inspired_exp6_causal_state_not_full_ros_lens_parity"
EXP6_ALPHAS: tuple[float, ...] = (0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035)
EXP6_INTERNAL_DT_MS = 0.2
EXP6_OUTPUT_INTERVAL_MS = 7.0
EXP6_OUTPUT_TIME_BINS = 35
EXP6_OFFICIAL_COMMIT_SHA = "59c498b71ae526bc2d7e570c82a078306a996b93"
EXP6_PROCESSOR_SOURCE_PATH = "ev_ttc/include/ev_ttc/ev_processor.h"
EXP6_PROCESSOR_SOURCE_SHA256 = "439384787969f36f72bdc72e3f6a058c33847f7f8a70454a44313ffc0e9d511e"
EXP6_CONFIG_SOURCE_PATH = "ev_ttc/include/ev_ttc/config.h"
EXP6_CONFIG_SOURCE_SHA256 = "d30bfe8b292cb8505b1e1841bb76ebbeb2e1f34b3dce13c85b383252d4a44fe7"

EXP6_RASTER_CONTRACT = {
    "source_scope": (
        "filter-state equation parity under frozen raster assumptions, not full "
        "ROS/lens/downsample parity"
    ),
    "scheduling": (
        "EV-TTC triggers from its first event when elapsed time is >=7 ms and can overshoot; "
        "V8 instead fixes boundaries at EventBatch.t_start_us+n*7000 us"
    ),
    "equation": "window_event(j) += polarity * alpha * (1-alpha)**(-j); snapshot *= (1-alpha)**35",
    "polarity": "input polarity is normalized to {-1,+1}; +1 is positive and -1 is negative",
    "timestamp_quantization": (
        "j=floor((timestamp_us-origin_us)/200), with origin_us=EventBatch.t_start_us"
    ),
    "boundary": (
        "at origin_us+n*7000 us, snapshot precedes events timestamped exactly at the boundary; "
        "those events enter next window at j=0"
    ),
    "initialization": "zero state at the stable per-sequence EventBatch.t_start_us origin",
    "reset": (
        "manual, sequence change, endpoint rollback, or resolution change clears state to zero"
    ),
    "warmup": "zero-state warmup is retained; no synthetic prehistory is injected",
    "snapshot": (
        "only exact 7 ms boundary snapshots have EV-TTC filter-state parity; non-boundary "
        "snapshots retain the causal local recurrence"
    ),
}

_RESET_REASON_CODES: dict[str, float] = {
    "none": 0.0,
    "manual": 1.0,
    "sequence_changed": 2.0,
    "endpoint_rollback": 3.0,
    "resolution_changed": 4.0,
}


@dataclass(frozen=True)
class TemporalRepresentationOutput:
    """One deterministic endpoint representation and its causal provenance."""

    tensor: torch.Tensor
    endpoint_us: int
    support_start_us: int
    support_end_us: int
    event_count: int
    finite: bool
    source: str
    diagnostics: dict[str, float]


@runtime_checkable
class TemporalEndpointRepresentation(Protocol):
    """Encode a causal event prefix at one temporal endpoint."""

    def encode(
        self,
        events: EventBatch,
        endpoint_us: int,
        roi: torch.Tensor,
    ) -> TemporalRepresentationOutput:
        """Return a representation supported entirely at or before ``endpoint_us``."""


@dataclass(frozen=True)
class ScientificRecoveryV8Batch:
    """V8-only temporal batch; its step count intentionally differs from V4."""

    representations: torch.Tensor
    endpoint_us: torch.Tensor
    token_id: list[str]
    sequence_id: list[str]
    track_id: list[str]
    target_ttc: torch.Tensor
    sample_weight: torch.Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject malformed tensors and metadata before a model sees the batch."""

        values = self.representations
        if values.ndim != 5:
            raise ValueError("representations must have shape [B,steps,channels,H,W]")
        batch_size, steps, channels, height, width = values.shape
        if batch_size <= 0 or steps not in {2, 3} or min(channels, height, width) <= 0:
            raise ValueError("representations require B>0, steps in {2,3}, and positive dimensions")
        if not torch.isfinite(values).all():
            raise ValueError("representations must be finite")
        if self.endpoint_us.shape != (batch_size, steps):
            raise ValueError("endpoint_us must have shape [B,steps]")
        if not torch.isfinite(self.endpoint_us).all():
            raise ValueError("endpoint_us must be finite")
        if bool((self.endpoint_us[:, 1:] < self.endpoint_us[:, :-1]).any()):
            raise ValueError("endpoint_us must be monotonic within every row")
        for name, identities in (
            ("token_id", self.token_id),
            ("sequence_id", self.sequence_id),
            ("track_id", self.track_id),
        ):
            if len(identities) != batch_size:
                raise ValueError(f"{name} length must equal batch size")
        for name, tensor in (
            ("target_ttc", self.target_ttc),
            ("sample_weight", self.sample_weight),
        ):
            if tensor.ndim == 0 or tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must have a leading batch dimension")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        for key, value in self.metadata.items():
            if isinstance(value, (list, tuple)) and len(value) != batch_size:
                raise ValueError(f"metadata[{key!r}] length must equal batch size")
            if isinstance(value, (torch.Tensor, np.ndarray)):
                if value.ndim == 0 or value.shape[0] != batch_size:
                    raise ValueError(f"metadata[{key!r}] must have a leading batch dimension")
                metadata_tensor = torch.as_tensor(value)
                if (
                    metadata_tensor.is_floating_point()
                    and not torch.isfinite(metadata_tensor).all()
                ):
                    raise ValueError(f"metadata[{key!r}] must be finite")


def _roi_as_int_tuple(roi: torch.Tensor) -> tuple[int, int, int, int]:
    """Validate a finite integer xyxy ROI and return it as Python integers."""

    if not isinstance(roi, torch.Tensor) or roi.numel() != 4:
        raise ValueError("roi must be a torch.Tensor with four xyxy coordinates")
    values = roi.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("roi must be finite")
    rounded = torch.round(values)
    if not torch.equal(values, rounded):
        raise ValueError("roi coordinates must be integer-valued")
    x_min, y_min, x_max, y_max = (int(value) for value in rounded.tolist())
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("roi must have positive width and height")
    return x_min, y_min, x_max, y_max


def _validate_events(events: EventBatch, endpoint_us: int) -> None:
    """Validate the real NumPy ``EventBatch`` contract without mutating it."""

    if not isinstance(events, EventBatch):
        raise TypeError("events must be an EventBatch")
    arrays = (events.x, events.y, events.t_us, events.polarity)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("EventBatch arrays must be one-dimensional")
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("EventBatch arrays must be aligned")
    if events.width <= 0 or events.height <= 0:
        raise ValueError("EventBatch resolution must be positive")
    timestamps = np.asarray(events.t_us, dtype=np.int64)
    if timestamps.size and np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("EventBatch timestamps must be monotonic")
    if timestamps.size and int(timestamps[-1]) > int(endpoint_us):
        raise ValueError("events after endpoint_us are not causal")
    if timestamps.size:
        x_values = np.asarray(events.x)
        y_values = np.asarray(events.y)
        if np.any(x_values < 0) or np.any(x_values >= events.width):
            raise ValueError("event x coordinates are outside the sensor")
        if np.any(y_values < 0) or np.any(y_values >= events.height):
            raise ValueError("event y coordinates are outside the sensor")


def _common_diagnostics(
    *,
    roi: tuple[int, int, int, int],
    target_size: tuple[int, int],
    event_count: int,
    support_start_us: int,
    support_end_us: int,
) -> dict[str, float]:
    return {
        "roi_xmin": float(roi[0]),
        "roi_ymin": float(roi[1]),
        "roi_xmax": float(roi[2]),
        "roi_ymax": float(roi[3]),
        "target_height": float(target_size[0]),
        "target_width": float(target_size[1]),
        "event_count": float(event_count),
        "support_start_us": float(support_start_us),
        "support_end_us": float(support_end_us),
    }


class GarlTimeVolumeRepresentation:
    """Causal 20-plane Garl-compatible time-volume frontend for one endpoint."""

    def __init__(
        self,
        *,
        window_ms: float = 100.0,
        number_of_planes: int = 20,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        if window_ms <= 0.0 or number_of_planes <= 0 or min(target_size) <= 0:
            raise ValueError("window_ms, number_of_planes, and target_size must be positive")
        self.window_ms = float(window_ms)
        self.number_of_planes = int(number_of_planes)
        self.target_size = (int(target_size[0]), int(target_size[1]))

    def encode(
        self,
        events: EventBatch,
        endpoint_us: int,
        roi: torch.Tensor,
    ) -> TemporalRepresentationOutput:
        """Encode the inclusive causal ``window_ms`` interval ending at endpoint."""

        endpoint = int(endpoint_us)
        _validate_events(events, endpoint)
        roi_xyxy = _roi_as_int_tuple(roi)
        support_start = endpoint - int(round(self.window_ms * 1_000.0))
        timestamps = np.asarray(events.t_us, dtype=np.int64)
        in_window = (timestamps >= support_start) & (timestamps <= endpoint)
        feature, counts = official_timevolume_roi_np(
            roi_xyxy,
            np.asarray(events.x)[in_window],
            np.asarray(events.y)[in_window],
            timestamps[in_window],
            time_window_s=self.window_ms / 1_000.0,
            number_of_planes=self.number_of_planes,
        )
        tensor = official_resize_feature(feature, self.target_size).to(dtype=torch.float32)
        event_count = int(counts.sum())
        finite = bool(torch.isfinite(tensor).all())
        if not finite:
            raise RuntimeError("official Garl time-volume preprocessing returned non-finite values")
        diagnostics = _common_diagnostics(
            roi=roi_xyxy,
            target_size=self.target_size,
            event_count=event_count,
            support_start_us=support_start,
            support_end_us=endpoint,
        )
        diagnostics.update(
            {
                "window_ms": self.window_ms,
                "number_of_planes": float(self.number_of_planes),
                "reset_count": 0.0,
                "reset_reason": _RESET_REASON_CODES["none"],
            }
        )
        return TemporalRepresentationOutput(
            tensor=tensor.contiguous(),
            endpoint_us=endpoint,
            support_start_us=support_start,
            support_end_us=endpoint,
            event_count=event_count,
            finite=finite,
            source=GARL_TIMEVOL_SOURCE,
            diagnostics=diagnostics,
        )


class CausalExponentialStateRepresentation:
    """Deterministic six-scale signed event state with causal snapshots.

    ``encode`` accepts causally complete packets or prefixes.  Re-supplying a
    prefix is safe: events at or before the latest ingested timestamp are not
    added twice.  Late events before an already snapshotted bin are rejected,
    because retroactive insertion would violate the explicit causal-state
    contract.
    """

    def __init__(
        self,
        *,
        alphas: tuple[float, ...] = EXP6_ALPHAS,
        internal_dt_ms: float = EXP6_INTERNAL_DT_MS,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        if not alphas or any(alpha <= 0.0 or alpha >= 1.0 for alpha in alphas):
            raise ValueError("alphas must be non-empty values strictly between zero and one")
        if internal_dt_ms <= 0.0 or min(target_size) <= 0:
            raise ValueError("internal_dt_ms and target_size must be positive")
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.internal_dt_ms = float(internal_dt_ms)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self._dt_us = int(round(self.internal_dt_ms * 1_000.0))
        if self._dt_us <= 0:
            raise ValueError("internal_dt_ms is too small to represent in microseconds")
        self._reset_count = 0
        self._last_reset_reason = "none"
        self._clear_state()

    @property
    def reset_count(self) -> int:
        """Return the total number of explicit or automatic state resets."""

        return self._reset_count

    @property
    def last_reset_reason(self) -> str:
        """Return the human-readable reason for the latest state reset."""

        return self._last_reset_reason

    def _clear_state(self) -> None:
        self._state: np.ndarray | None = None
        self._sequence_id: str | None = None
        self._resolution: tuple[int, int] | None = None
        self._origin_us: int | None = None
        self._last_bin = -1
        self._last_endpoint_us: int | None = None
        self._last_ingested_t_us: int | None = None
        self._state_event_count = 0
        self._pending_boundary_events: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pending_boundary_endpoint_us: int | None = None

    def _is_output_boundary(self, endpoint_us: int) -> bool:
        """Return whether an endpoint is an exact frozen 7 ms snapshot boundary."""

        if self._origin_us is None:
            return False
        output_interval_us = int(round(EXP6_OUTPUT_INTERVAL_MS * 1_000.0))
        return (
            endpoint_us > self._origin_us
            and (endpoint_us - self._origin_us) % output_interval_us == 0
        )

    def _commit_pending_boundary_events(self, next_endpoint_us: int) -> None:
        """Commit a boundary event when, and only when, the next update advances."""

        if self._pending_boundary_events is None:
            return
        if self._pending_boundary_endpoint_us is None:
            raise RuntimeError("pending EXP6 boundary events have no endpoint provenance")
        if next_endpoint_us <= self._pending_boundary_endpoint_us:
            raise RuntimeError("must advance beyond the pending EXP6 boundary before updating")
        x_values, y_values, polarity_values = self._pending_boundary_events
        self._apply_bin(
            self._bin_for_timestamp(self._pending_boundary_endpoint_us),
            x_values,
            y_values,
            polarity_values,
        )
        self._state_event_count += int(x_values.size)
        self._pending_boundary_events = None
        self._pending_boundary_endpoint_us = None

    def reset(self, *, reason: str = "manual") -> None:
        """Discard all state and record a reset reason for the next snapshot."""

        if reason not in _RESET_REASON_CODES:
            raise ValueError(f"unknown reset reason: {reason}")
        self._clear_state()
        self._reset_count += 1
        self._last_reset_reason = reason

    def _ensure_sequence(self, events: EventBatch, endpoint_us: int) -> None:
        resolution = (int(events.height), int(events.width))
        if self._sequence_id is not None and events.sequence_id != self._sequence_id:
            self.reset(reason="sequence_changed")
        elif self._last_endpoint_us is not None and endpoint_us < self._last_endpoint_us:
            self.reset(reason="endpoint_rollback")
        elif self._resolution is not None and resolution != self._resolution:
            self.reset(reason="resolution_changed")
        if self._state is None:
            self._sequence_id = events.sequence_id
            self._resolution = resolution
            self._origin_us = int(events.t_start_us)
            if endpoint_us < self._origin_us:
                raise ValueError("endpoint_us precedes the stable per-sequence origin")
            self._state = np.zeros((len(self.alphas), *resolution), dtype=np.float32)

    def _bin_for_timestamp(self, timestamp_us: int) -> int:
        if self._origin_us is None:
            raise RuntimeError("state origin is not initialized")
        if timestamp_us < self._origin_us:
            raise ValueError("event timestamp precedes the stable per-sequence origin")
        return (int(timestamp_us) - self._origin_us) // self._dt_us

    def _advance_empty_bins(self, target_bin: int) -> None:
        if self._state is None:
            raise RuntimeError("state is not initialized")
        if target_bin <= self._last_bin:
            return
        bin_count = target_bin - self._last_bin
        decays = np.power(
            1.0 - np.asarray(self.alphas, dtype=np.float32), np.float32(bin_count)
        ).astype(np.float32)
        self._state *= decays[:, None, None]
        self._last_bin = target_bin

    def _apply_bin(
        self, bin_index: int, x: np.ndarray, y: np.ndarray, polarity: np.ndarray
    ) -> None:
        if self._state is None:
            raise RuntimeError("state is not initialized")
        if bin_index < self._last_bin:
            raise ValueError("cannot insert events into an already snapshotted bin")
        if bin_index > self._last_bin:
            self._advance_empty_bins(bin_index)
        else:
            # The bin has not yet received its mandatory decay when starting from -1.
            if self._last_bin == -1:
                self._advance_empty_bins(bin_index)
        signed_counts = np.zeros(self._state.shape[1:], dtype=np.float32)
        np.add.at(
            signed_counts, (y.astype(np.intp), x.astype(np.intp)), polarity.astype(np.float32)
        )
        alpha = np.asarray(self.alphas, dtype=np.float32)[:, None, None]
        self._state += alpha * signed_counts[None]

    def update(self, events: EventBatch, endpoint_us: int) -> int:
        """Incrementally ingest a causal packet and advance state to endpoint.

        Returns the number of newly ingested events.  Events beyond endpoint are
        rejected before state is changed.
        """

        endpoint = int(endpoint_us)
        _validate_events(events, endpoint)
        polarity = np.asarray(events.polarity)
        if polarity.size and not np.isin(polarity, (-1, 1)).all():
            raise ValueError("EXP6 requires polarities normalized to {-1, +1}")
        self._ensure_sequence(events, endpoint)
        self._commit_pending_boundary_events(endpoint)
        timestamps = np.asarray(events.t_us, dtype=np.int64)
        new_mask = (
            np.ones(timestamps.shape, dtype=bool)
            if self._last_ingested_t_us is None
            else timestamps > self._last_ingested_t_us
        )
        x_values = np.asarray(events.x)[new_mask]
        y_values = np.asarray(events.y)[new_mask]
        time_values = timestamps[new_mask]
        polarity_values = polarity[new_mask]
        boundary_mask = np.zeros(time_values.shape, dtype=bool)
        if self._is_output_boundary(endpoint):
            boundary_mask = time_values == endpoint
        pending_count = int(np.count_nonzero(boundary_mask))
        ingest_mask = ~boundary_mask
        ingest_x = x_values[ingest_mask]
        ingest_y = y_values[ingest_mask]
        ingest_time = time_values[ingest_mask]
        ingest_polarity = polarity_values[ingest_mask]
        if ingest_time.size:
            bins = np.asarray([self._bin_for_timestamp(int(value)) for value in ingest_time])
            if bins[0] < self._last_bin:
                raise ValueError("new events precede the already snapshotted causal state")
            starts = np.r_[0, np.flatnonzero(bins[1:] != bins[:-1]) + 1]
            ends = np.r_[starts[1:], len(bins)]
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                self._apply_bin(
                    int(bins[start]),
                    ingest_x[start:end],
                    ingest_y[start:end],
                    ingest_polarity[start:end],
                )
            self._state_event_count += int(ingest_time.size)
        if time_values.size:
            self._last_ingested_t_us = int(time_values[-1])
        endpoint_bin = self._bin_for_timestamp(endpoint)
        self._advance_empty_bins(endpoint_bin)
        if pending_count:
            self._pending_boundary_events = (
                x_values[boundary_mask].copy(),
                y_values[boundary_mask].copy(),
                polarity_values[boundary_mask].copy(),
            )
            self._pending_boundary_endpoint_us = endpoint
        self._last_endpoint_us = endpoint
        return int(ingest_time.size)

    def snapshot(
        self, endpoint_us: int, roi: torch.Tensor, *, event_count: int = 0
    ) -> TemporalRepresentationOutput:
        """Resize the current causal state at an already advanced endpoint."""

        endpoint = int(endpoint_us)
        if self._state is None or self._origin_us is None or self._last_endpoint_us is None:
            raise RuntimeError("update must be called before snapshot")
        if endpoint != self._last_endpoint_us:
            raise ValueError("snapshot endpoint must equal the latest advanced endpoint")
        roi_xyxy = _roi_as_int_tuple(roi)
        x_min, y_min, x_max, y_max = roi_xyxy
        cropped = np.zeros((len(self.alphas), y_max - y_min, x_max - x_min), dtype=np.float32)
        source_y0 = max(y_min, 0)
        source_y1 = min(y_max, self._state.shape[1])
        source_x0 = max(x_min, 0)
        source_x1 = min(x_max, self._state.shape[2])
        if source_y1 > source_y0 and source_x1 > source_x0:
            cropped[
                :, source_y0 - y_min : source_y1 - y_min, source_x0 - x_min : source_x1 - x_min
            ] = self._state[:, source_y0:source_y1, source_x0:source_x1]
        tensor = official_resize_feature(cropped, self.target_size).to(dtype=torch.float32)
        finite = bool(torch.isfinite(tensor).all())
        if not finite:
            raise RuntimeError("EXP6 state snapshot returned non-finite values")
        diagnostics = _common_diagnostics(
            roi=roi_xyxy,
            target_size=self.target_size,
            event_count=event_count,
            support_start_us=self._origin_us,
            support_end_us=endpoint,
        )
        diagnostics.update(
            {
                "reset_count": float(self._reset_count),
                "reset_reason": _RESET_REASON_CODES[self._last_reset_reason],
                "warmup_duration_us": float(endpoint - self._origin_us),
                "internal_dt_ms": self.internal_dt_ms,
                "state_event_count": float(self._state_event_count),
                **{f"alpha_{index}": alpha for index, alpha in enumerate(self.alphas)},
            }
        )
        output = TemporalRepresentationOutput(
            tensor=tensor.contiguous(),
            endpoint_us=endpoint,
            support_start_us=self._origin_us,
            support_end_us=endpoint,
            event_count=int(event_count),
            finite=finite,
            source=EXP6_SOURCE,
            diagnostics=diagnostics,
        )
        return output

    def encode(
        self,
        events: EventBatch,
        endpoint_us: int,
        roi: torch.Tensor,
    ) -> TemporalRepresentationOutput:
        """Update from a causal packet/prefix and return its EXP6 snapshot."""

        event_count = self.update(events, endpoint_us)
        return self.snapshot(endpoint_us, roi, event_count=event_count)


__all__ = [
    "EXP6_ALPHAS",
    "EXP6_CONFIG_SOURCE_PATH",
    "EXP6_CONFIG_SOURCE_SHA256",
    "EXP6_INTERNAL_DT_MS",
    "EXP6_OFFICIAL_COMMIT_SHA",
    "EXP6_OUTPUT_INTERVAL_MS",
    "EXP6_OUTPUT_TIME_BINS",
    "EXP6_PROCESSOR_SOURCE_PATH",
    "EXP6_PROCESSOR_SOURCE_SHA256",
    "EXP6_RASTER_CONTRACT",
    "EXP6_SOURCE",
    "GARL_TIMEVOL_SOURCE",
    "CausalExponentialStateRepresentation",
    "GarlTimeVolumeRepresentation",
    "ScientificRecoveryV8Batch",
    "TemporalEndpointRepresentation",
    "TemporalRepresentationOutput",
]
