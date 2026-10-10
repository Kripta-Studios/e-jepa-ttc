"""Connect V13 causal observations to lossless packets and reusable sensor crops."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.streaming_revision.packets import PacketRing
from operational.streaming_revision.preparation import IncrementalPreparer, Query
from operational.streaming_revision.state import roi_distance

from .runtime import Request


class EventInputs:
    """Cache event voxels, optionally reprojecting only original historical voxels.

    Pass actual V13 observations; this class never manufactures a 50ms history.
    Reset both input and feature caches on a sensor restart.
    """

    def __init__(
        self, ring: PacketRing, *, approximate_voxels: bool = False, offset: float = 0.0
    ) -> None:
        self.ring, self.offset = ring, offset
        self.preparer = IncrementalPreparer(
            ring.read_window, roi_reader=ring.read_roi, approximate_voxels=approximate_voxels
        )

    def request(
        self,
        sequence: str,
        track: str,
        now: int,
        *,
        anchor: int,
        available: int,
        windows: Sequence[tuple[int, int]],
        roi: tuple[float, ...],
    ) -> Request:
        """Retain the observation's true interval and ROI availability identity."""
        windows = tuple(windows)
        delta = tuple((b[1] - a[1]) / 1e6 for a, b in zip(windows, windows[1:], strict=False))
        query = Query(
            sequence, track, anchor, available, windows, roi, delta=delta[-1], offset=self.offset
        )
        request = Request(
            anchor,
            available,
            windows,
            roi,
            delta,
            lambda: self.preparer.prepare(query, [anchor])[0],
        )
        request.validate(now)
        return request


class RGBInputs:
    """Exact bounded uint8 frame cache; each requested ROI uses native cropping."""

    def __init__(
        self, reader: Callable[[str, int], np.ndarray], *, max_bytes: int = 64 * 1024**2
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("Positive RGB cache bound required")
        self.reader, self.max_bytes = reader, max_bytes
        self.frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self.retained_bytes = 0

    def reset(self) -> None:
        self.frames.clear()
        self.retained_bytes = 0

    def _frame(self, sequence: str, timestamp: int) -> np.ndarray:
        key = sequence, timestamp
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        frame = self.reader(*key)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("RGB reader must return native uint8 HWC frames")
        frame = frame.copy()
        if frame.nbytes <= self.max_bytes:
            while self.frames and self.retained_bytes + frame.nbytes > self.max_bytes:
                _, removed = self.frames.popitem(last=False)
                self.retained_bytes -= removed.nbytes
            self.frames[key] = frame
            self.retained_bytes += frame.nbytes
        return frame

    def request(
        self,
        sequence: str,
        now: int,
        *,
        timestamps: Sequence[int],
        available: int,
        roi: tuple[float, ...],
    ) -> Request:
        """Use two or three real sensor frames and measured deltas, without padding."""
        times = tuple(timestamps)
        if len(times) not in (2, 3) or any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Ordered real RGB T2/T3 timestamps required")
        roi_distance(roi, roi)

        def load() -> torch.Tensor:
            crops = [crop_uint8(self._frame(sequence, t), roi) for t in times]
            return torch.from_numpy(np.stack(crops).astype(np.float32) / 255)

        request = Request(
            times[-1],
            available,
            tuple(zip(times, times[1:], strict=False)),
            roi,
            tuple((b - a) / 1e6 for a, b in zip(times, times[1:], strict=False)),
            load,
        )
        request.validate(now)
        return request
