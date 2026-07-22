"""Bounded-memory streaming inference for object-ROI event streams."""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass

import numpy as np
import torch

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    inverse_ttc_distribution_to_seconds,
)
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid


@dataclass(frozen=True)
class _ObjectObservation:
    timestamp_us: int
    box_xyxy: np.ndarray
    ego_action: np.ndarray
    ego_action_valid: bool


@dataclass(frozen=True)
class StreamingPrediction:
    """One probabilistic TTC result with separated latency measurements."""

    timestamp_us: int
    ttc_mean_s: float
    ttc_std_s: float
    risk_probabilities: tuple[float, ...]
    risk_state: str
    preprocessing_ms: float
    inference_ms: float
    event_count: int
    event_rate_hz: float


class StreamingTTCEstimator:
    """Infer TTC from a bounded ring buffer for one detector-tracked object.

    Incoming event coordinates are ROI-local. The associated global normalized
    object box is supplied separately with :meth:`push_observation`.
    """

    def __init__(
        self,
        model: ObjectCentricEventJEPA,
        *,
        width: int = 64,
        height: int = 64,
        event_bins: int = 5,
        event_window_ms: int = 100,
        history_steps: int = 3,
        observation_slop_ms: int = 125,
        minimum_events: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(width, height, event_bins, event_window_ms, history_steps) <= 0:
            msg = "Streaming dimensions, bins, windows and history must be positive."
            raise ValueError(msg)
        if observation_slop_ms < 0 or minimum_events < 0:
            msg = "observation_slop_ms and minimum_events must be non-negative."
            raise ValueError(msg)
        expected_channels = event_bins * 2
        if model.config.in_channels != expected_channels:
            msg = (
                f"Model expects {model.config.in_channels} channels but streaming "
                f"configuration produces {expected_channels}."
            )
            raise ValueError(msg)
        self.model = model
        self.width = width
        self.height = height
        self.event_bins = event_bins
        self.event_window_us = event_window_ms * 1000
        self.history_steps = history_steps
        self.observation_slop_us = observation_slop_ms * 1000
        self.minimum_events = minimum_events
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.reset()

    def reset(self) -> None:
        """Clear events, observations and timestamp state."""

        self._x = np.empty(0, dtype=np.int32)
        self._y = np.empty(0, dtype=np.int32)
        self._t_us = np.empty(0, dtype=np.int64)
        self._polarity = np.empty(0, dtype=np.int8)
        self._observations: list[_ObjectObservation] = []
        self._last_timestamp_us: int | None = None

    def push_events(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t_us: np.ndarray,
        polarity: np.ndarray,
    ) -> None:
        """Append a monotonic packet and discard data older than bounded history."""

        arrays = [np.asarray(value).reshape(-1) for value in (x, y, t_us, polarity)]
        if len({array.size for array in arrays}) != 1:
            msg = "Streaming event packet arrays must have equal length."
            raise ValueError(msg)
        if arrays[0].size == 0:
            return
        packet_x = arrays[0].astype(np.int32, copy=False)
        packet_y = arrays[1].astype(np.int32, copy=False)
        packet_t = arrays[2].astype(np.int64, copy=False)
        packet_p = arrays[3]
        if np.any(np.diff(packet_t) < 0):
            msg = "Streaming event packet timestamps must be monotonic."
            raise ValueError(msg)
        if self._last_timestamp_us is not None and int(packet_t[0]) < self._last_timestamp_us:
            msg = "Timestamp rollback detected; call reset() before starting a new stream."
            raise ValueError(msg)
        if np.any((packet_x < 0) | (packet_x >= self.width)):
            msg = "Streaming event x coordinates are outside the ROI."
            raise ValueError(msg)
        if np.any((packet_y < 0) | (packet_y >= self.height)):
            msg = "Streaming event y coordinates are outside the ROI."
            raise ValueError(msg)
        packet_p = np.where(packet_p > 0, 1, -1).astype(np.int8)
        self._x = np.concatenate((self._x, packet_x))
        self._y = np.concatenate((self._y, packet_y))
        self._t_us = np.concatenate((self._t_us, packet_t))
        self._polarity = np.concatenate((self._polarity, packet_p))
        self._last_timestamp_us = int(packet_t[-1])
        self._trim(self._last_timestamp_us)

    def push_observation(
        self,
        timestamp_us: int,
        box_xyxy: np.ndarray,
        *,
        ego_action: np.ndarray | None = None,
        ego_action_valid: bool = False,
    ) -> None:
        """Append one causal detector box and optional measured egoaction."""

        box = np.asarray(box_xyxy, dtype=np.float32)
        if box.shape != (4,) or not np.all(np.isfinite(box)):
            msg = "box_xyxy must be a finite normalized four-vector."
            raise ValueError(msg)
        if np.any((box < 0) | (box > 1)) or box[2] <= box[0] or box[3] <= box[1]:
            msg = "box_xyxy must be a non-empty normalized xyxy box."
            raise ValueError(msg)
        action = np.zeros(self.model.config.action_dim, dtype=np.float32)
        if ego_action is not None:
            action = np.asarray(ego_action, dtype=np.float32)
            if action.shape != (self.model.config.action_dim,) or not np.all(np.isfinite(action)):
                msg = "ego_action has an incompatible shape or non-finite values."
                raise ValueError(msg)
        if self._observations and timestamp_us < self._observations[-1].timestamp_us:
            msg = "Object observations must be appended in timestamp order."
            raise ValueError(msg)
        self._observations.append(
            _ObjectObservation(
                timestamp_us=int(timestamp_us),
                box_xyxy=box.copy(),
                ego_action=action.copy(),
                ego_action_valid=bool(ego_action_valid),
            )
        )
        self._trim(int(timestamp_us))

    def ready(self, now_us: int | None = None) -> bool:
        """Return whether bounded history and causal boxes support a prediction."""

        if now_us is None:
            if self._last_timestamp_us is None:
                return False
            now_us = self._last_timestamp_us + 1
        oldest = int(now_us) - self.history_steps * self.event_window_us
        if self._t_us.size == 0 or int(self._t_us[0]) > oldest:
            return False
        return all(
            self._observation_for(endpoint) is not None for endpoint in self._endpoints(int(now_us))
        )

    def predict(self, now_us: int) -> StreamingPrediction:
        """Predict from exactly the retained history, never from future packets."""

        if not self.ready(now_us):
            msg = "Streaming estimator is not ready at the requested timestamp."
            raise RuntimeError(msg)
        preprocessing_start = time.perf_counter()
        frames: list[np.ndarray] = []
        boxes: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        action_masks: list[bool] = []
        event_count = 0
        for endpoint in self._endpoints(now_us):
            start = endpoint - self.event_window_us
            begin = int(np.searchsorted(self._t_us, start, side="left"))
            stop = int(np.searchsorted(self._t_us, endpoint, side="left"))
            batch = EventBatch(
                x=self._x[begin:stop],
                y=self._y[begin:stop],
                t_us=self._t_us[begin:stop],
                polarity=self._polarity[begin:stop],
                width=self.width,
                height=self.height,
                sequence_id="stream",
                t_start_us=start,
                t_end_us=endpoint,
            )
            event_count += batch.num_events
            frames.append(
                encode_voxel_grid(
                    batch,
                    bins=self.event_bins,
                    separate_polarity=True,
                    normalize=True,
                )
            )
            observation = self._observation_for(endpoint)
            if observation is None:
                msg = "Object observation became unavailable during streaming preprocessing."
                raise RuntimeError(msg)
            boxes.append(observation.box_xyxy)
            actions.append(observation.ego_action)
            action_masks.append(observation.ego_action_valid)
        if event_count < self.minimum_events:
            msg = "Streaming history contains fewer than minimum_events."
            raise RuntimeError(msg)
        context_events = torch.from_numpy(np.stack(frames))[None].to(self.device)
        context_boxes = torch.from_numpy(np.stack(boxes))[:, None, :][None].to(self.device)
        object_mask = torch.ones((1, self.history_steps, 1), dtype=torch.bool, device=self.device)
        sampling_boxes = (
            torch.tensor(
                [0.0, 0.0, 1.0, 1.0],
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(1, 1, 1, 4)
            .expand(1, self.history_steps, 1, 4)
        )
        ego_actions = torch.from_numpy(np.stack(actions))[None].to(self.device)
        action_mask = torch.tensor(action_masks, dtype=torch.bool, device=self.device)[None]
        preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0

        self._synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.predict_ttc(
                context_events,
                context_boxes,
                object_mask,
                context_sampling_boxes=sampling_boxes,
                context_ego_actions=ego_actions,
                context_ego_action_mask=action_mask,
            )
            ttc_mean, ttc_std = inverse_ttc_distribution_to_seconds(
                output.inverse_ttc_mean,
                output.inverse_ttc_log_variance,
            )
            risk = torch.sigmoid(output.risk_logits)
        self._synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        probabilities = tuple(float(value) for value in risk[0, 0].cpu())
        mean = float(ttc_mean[0, 0].cpu())
        std = float(ttc_std[0, 0].cpu())
        return StreamingPrediction(
            timestamp_us=int(now_us),
            ttc_mean_s=mean,
            ttc_std_s=std,
            risk_probabilities=probabilities,
            risk_state=self._risk_state(mean, probabilities),
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            event_count=event_count,
            event_rate_hz=event_count / (self.history_steps * self.event_window_us * 1e-6),
        )

    def _endpoints(self, now_us: int) -> list[int]:
        return [
            now_us - (self.history_steps - 1 - index) * self.event_window_us
            for index in range(self.history_steps)
        ]

    def _observation_for(self, endpoint_us: int) -> _ObjectObservation | None:
        timestamps = [value.timestamp_us for value in self._observations]
        index = bisect.bisect_right(timestamps, endpoint_us) - 1
        if index < 0:
            return None
        observation = self._observations[index]
        if endpoint_us - observation.timestamp_us > self.observation_slop_us:
            return None
        return observation

    def _trim(self, now_us: int) -> None:
        keep_after = now_us - self.history_steps * self.event_window_us
        start = int(np.searchsorted(self._t_us, keep_after, side="left"))
        if start:
            self._x = self._x[start:]
            self._y = self._y[start:]
            self._t_us = self._t_us[start:]
            self._polarity = self._polarity[start:]
        observation_keep_after = keep_after - self.observation_slop_us
        first = bisect.bisect_left(
            [value.timestamp_us for value in self._observations],
            observation_keep_after,
        )
        if first:
            self._observations = self._observations[first:]

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _risk_state(self, ttc_mean_s: float, probabilities: tuple[float, ...]) -> str:
        if not np.isfinite(ttc_mean_s) or not probabilities:
            return "UNKNOWN"
        active = [
            threshold
            for threshold, probability in zip(
                self.model.config.risk_thresholds_s,
                probabilities,
                strict=True,
            )
            if probability >= 0.5
        ]
        if not active:
            return "SAFE"
        shortest = min(active)
        if shortest <= 0.5:
            return "CRITICAL"
        if shortest <= 1.0:
            return "WARNING"
        return "WATCH"


__all__ = ["StreamingPrediction", "StreamingTTCEstimator"]
