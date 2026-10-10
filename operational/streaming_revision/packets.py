"""Bounded live packet ring with explicit coverage, gap and rollback handling."""

from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np

from .preparation import Raw


def _lower_bound(times: np.ndarray, value: int) -> int:
    """Search without array promotion, including coverage outside the input dtype."""
    if not len(times) or value <= int(times[0]):
        return 0
    if value > int(times[-1]):
        return len(times)
    return int(np.searchsorted(times, np.asarray(value, dtype=times.dtype)))


def packet_bounds(
    times: np.ndarray, first: int, last: int, start: int, end: int
) -> tuple[int, int]:
    """Search offsets without promoting and copying the entire uint32 time column."""
    lo = (
        0
        if start <= first
        else int(np.searchsorted(times, np.asarray(start - first, dtype=times.dtype)))
    )
    hi = (
        len(times)
        if end >= last
        else int(np.searchsorted(times, np.asarray(end - first, dtype=times.dtype)))
    )
    return lo, hi


class PacketRing:
    """Lossless compact packets; empty packets still advance known sensor coverage.

    Timestamps use unsigned offsets within each packet, coordinates use int16
    when representable. Reads restore the public int32/int64/int8 contract.
    """

    def __init__(
        self,
        *,
        horizon_us: int = 1000000,
        max_bytes: int = 256 * 1024**2,
        packet_span_us: int = 0,
    ) -> None:
        if horizon_us < 650000 or max_bytes <= 0:
            raise ValueError("at least 650 ms coverage and a positive memory limit required")
        self.horizon_us, self.max_bytes = horizon_us, max_bytes
        if packet_span_us < 0:
            raise ValueError("packet span must be nonnegative; zero preserves incoming packets")
        self.packet_span_us = packet_span_us
        self.packets: deque[tuple[int, int, Raw]] = deque()
        self.start: int | None = None
        self.end: int | None = None
        self.retained_bytes = 0

    def reset(self) -> None:
        """Reset explicitly after sensor restart or failed memory admission."""
        self.packets.clear()
        self.start, self.end, self.retained_bytes = None, None, 0

    def push(self, raw: Raw, start: int, end: int) -> None:
        """Append a contiguous half-open packet; no silent data loss or timestamp repair."""
        if (
            start < 0
            or end <= start
            or end > np.iinfo(np.int64).max
            or (self.end is not None and start != self.end)
        ):
            raise ValueError("packet gap or rollback: reset and refill the stream")
        if set(raw) != {"x", "y", "t", "p"}:
            raise ValueError("four event columns required")
        times = raw["t"]
        if times.dtype.kind not in "iu" or times.ndim != 1:
            raise ValueError("integer microsecond timestamps required")
        if any(v.shape != times.shape for v in raw.values()):
            raise ValueError("unaligned event columns")
        if len(times) and (
            int(times[0]) < start or int(times[-1]) >= end or (times[1:] < times[:-1]).any()
        ):
            raise ValueError("packet events violate timestamp order/coverage")
        if any(raw[k].dtype.kind not in "iub" for k in ("x", "y", "p")):
            raise ValueError("integer coordinates and polarity required")
        ranges = {
            k: (int(raw[k].min()), int(raw[k].max())) if len(times) else (0, 0) for k in ("x", "y")
        }
        for key, lower, upper in (
            ("x", -(2**31), 2**31 - 1),
            ("y", -(2**31), 2**31 - 1),
            ("p", -128, 127),
        ):
            if key == "p":
                bounds = np.iinfo(raw[key].dtype) if raw[key].dtype.kind != "b" else None
                fits = bounds is None or (bounds.min >= lower and bounds.max <= upper)
                limits = (
                    (0, 1) if fits or not len(times) else (int(raw[key].min()), int(raw[key].max()))
                )
            else:
                limits = ranges[key]
            if limits[0] < lower or limits[1] > upper:
                raise ValueError("event values exceed the public integer representation")
        cutoff = end - self.horizon_us
        while self.packets and self.packets[0][1] <= cutoff:
            _, _, expired = self.packets.popleft()
            self.retained_bytes -= sum(v.nbytes for v in expired.values())
        if self.packets and self.packets[0][0] < cutoff:
            first, last, old = self.packets.popleft()
            self.retained_bytes -= sum(v.nbytes for v in old.values())
            index = int(np.searchsorted(old["t"], np.asarray(cutoff - first, dtype=old["t"].dtype)))
            trimmed = {k: v[index:].copy() for k, v in old.items()}
            trimmed["t"] -= cutoff - first
            self.packets.appendleft((cutoff, last, trimmed))
            self.retained_bytes += sum(v.nbytes for v in trimmed.values())
        retained_start = max(start, cutoff)
        span = self.packet_span_us or end - retained_start
        cursor = _lower_bound(times, retained_start)
        for first in range(retained_start, end, span):
            last = min(first + span, end)
            stop = len(times) if last == end else _lower_bound(times, last)
            owned: Raw = {"p": raw["p"][cursor:stop].astype(np.int8, copy=True)}
            for key in ("x", "y"):
                compact = ranges[key][0] >= -32768 and ranges[key][1] <= 32767
                owned[key] = raw[key][cursor:stop].astype(
                    np.int16 if compact else np.int32, copy=True
                )
            if last - first <= 2**32:
                # Modular subtraction is exact for these bounded offsets, even
                # when absolute timestamps cross a uint32 wrap boundary.
                owned["t"] = times[cursor:stop].astype(np.uint32)
                owned["t"] -= np.uint32(first % 2**32)
            else:
                owned["t"] = times[cursor:stop].astype(np.int64, copy=False) - first
            cost = sum(v.nbytes for v in owned.values())
            if self.retained_bytes + cost > self.max_bytes:
                self.reset()
                raise MemoryError("live packet memory limit exceeded; state discarded explicitly")
            self.packets.append((first, last, owned))
            self.retained_bytes += cost
            cursor = stop
        self.start = max(cutoff, self.start if self.start is not None else start)
        self.end = end

    def read_window(self, start: int, end: int) -> Raw:
        """Return only covered events, never a silent incomplete history."""
        if self.start is None or self.end is None or not self.start <= start < end <= self.end:
            raise ValueError("requested interval is not fully covered by the live ring")
        pieces = {k: [] for k in ("x", "y", "t", "p")}
        for first, last, raw in self.packets:
            if last <= start or first >= end:
                continue
            lo, hi = packet_bounds(raw["t"], first, last, start, end)
            for key in pieces:
                value = raw[key][lo:hi]
                pieces[key].append(value.astype(np.int64) + first if key == "t" else value)
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        return {
            k: np.concatenate(v).astype(dtypes[k], copy=False)
            if v
            else np.empty(0, dtype=dtypes[k])
            for k, v in pieces.items()
        }

    def read_roi(
        self,
        start: int,
        end: int,
        roi: tuple[float, ...],
        offset: float,
        *,
        preserve_sensor_origin: bool = False,
    ) -> Raw:
        """Crop compact packets before decompression; preserve chronological order.

        Garl uses the first sensor event as its time origin. Its caller must
        retain that event even when it is outside the requested object ROI.
        """
        if self.start is None or self.end is None or not self.start <= start < end <= self.end:
            raise ValueError("requested interval is not fully covered by the live ring")
        x0, y0, x1, y1 = roi
        if not np.isfinite([*roi, offset]).all() or x0 >= x1 or y0 >= y1:
            raise ValueError("finite positive ROI required")
        pieces: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "t", "p")}
        origin_kept = False
        for first, last, raw in self.packets:
            if last <= start or first >= end:
                continue
            lo, hi = packet_bounds(raw["t"], first, last, start, end)
            x, y = raw["x"][lo:hi], raw["y"][lo:hi]
            keep = (
                (x >= ceil(max(0, x0) - offset))
                & (x < ceil(min(1280, x1) - offset))
                & (y >= ceil(max(0, y0)))
                & (y < ceil(min(720, y1)))
            )
            if preserve_sensor_origin and not origin_kept:
                sensor = (x >= ceil(-offset)) & (x < ceil(1280 - offset)) & (y >= 0) & (y < 720)
                if sensor.any():
                    keep[int(sensor.argmax())] = True
                    origin_kept = True
            for key in pieces:
                values = raw[key][lo:hi][keep]
                pieces[key].append(values.astype(np.int64) + first if key == "t" else values)
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        return {
            k: np.concatenate(v).astype(dtypes[k], copy=False)
            if v
            else np.empty(0, dtype=dtypes[k])
            for k, v in pieces.items()
        }


def native_live_window(
    ring: PacketRing,
    start: int,
    end: int,
    roi: tuple[float, ...] | None = None,
) -> Raw:
    """Preserve native Garl floor-ms bounds, int16 offset order and sensor clipping."""
    first, last = (start // 1000) * 1000, (end // 1000) * 1000
    raw = (
        ring.read_window(first, last)
        if roi is None
        else ring.read_roi(first, last, roi, 5.0, preserve_sensor_origin=True)
    )
    x = (raw["x"] + 5).astype(np.int16)
    y = raw["y"].astype(np.int16)
    keep = (x >= 0) & (x < 1280) & (y >= 0) & (y < 720)
    return {"x": x[keep], "y": y[keep], "t": raw["t"][keep]}
