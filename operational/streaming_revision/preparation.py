"""Prepare only missing triplets from a bounded raw-event ring and voxel cache."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi

Raw = dict[str, np.ndarray]


def warp_voxel(value: Tensor, source: tuple[float, ...], target: tuple[float, ...]) -> Tensor:
    """Approximate ROI reprojection; preserve scalar planes, zero newly uncovered pixels.

    This cannot recover discarded events or exact normalization. Always resample
    the original encoded voxel, never a previously warped result.
    """
    if value.shape != (12, 128, 128):
        raise ValueError("expected a 12-channel 128px voxel")
    if source == target:
        return value
    position = (torch.arange(128, dtype=value.dtype, device=value.device) + 0.5) / 128
    y, x = torch.meshgrid(position, position, indexing="ij")
    x = 2 * (target[0] + x * (target[2] - target[0]) - source[0]) / (source[2] - source[0]) - 1
    y = 2 * (target[1] + y * (target[3] - target[1]) - source[1]) / (source[3] - source[1]) - 1
    grid = torch.stack((x, y), -1)[None]
    warped = functional.grid_sample(
        value[None, :10], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    return torch.cat((warped, value[10:]), 0)


@dataclass(frozen=True)
class Query:
    """Explicit stream identity and causal metadata availability for one query."""

    sequence: str
    track: str
    anchor: int
    available: int
    windows: tuple[tuple[int, int], ...]
    roi: tuple[float, ...]
    delta: float = 0.1
    offset: float = 0.0
    valid: tuple[bool, ...] = (True,) * 8

    def __post_init__(self) -> None:
        from .state import roi_distance

        roi_distance(self.roi, self.roi)
        w = np.asarray(self.windows)
        if w.shape != (3, 2) or (w[:, 1] <= w[:, 0]).any() or w.max() != self.anchor:
            raise ValueError("three positive causal endpoint windows required")
        if (np.diff(w[:, 1]) <= 0).any() or self.available < self.anchor:
            raise ValueError("invalid window order or query availability")
        if (
            len(self.valid) != 8
            or not self.valid[-1]
            or any(a and not b for a, b in zip(self.valid, self.valid[1:], strict=False))
        ):
            raise ValueError("eight-slot contiguous valid suffix required")
        if not self.sequence or not self.track or not np.isfinite([self.delta, self.offset]).all():
            raise ValueError("explicit stream identity and finite timing required")
        if self.delta <= 0:
            raise ValueError("positive delta required")

    def at(self, anchor: int) -> tuple[tuple[int, int], ...]:
        """Translate all endpoint bounds without rounding their measured durations."""
        return tuple((a + anchor - self.anchor, b + anchor - self.anchor) for a, b in self.windows)


class IncrementalPreparer:
    """Reader callback may receive live buffered packets or HDF5 slices."""

    def __init__(
        self,
        reader: Callable[[int, int], Raw],
        *,
        raw_bytes: int = 512 * 1024**2,
        voxel_bytes: int = 64 * 1024**2,
        approximate_voxels: bool = False,
        time_tolerance_us: int = 2500,
        roi_reader: Callable[[int, int, tuple[float, ...], float], Raw] | None = None,
    ) -> None:
        if raw_bytes < 0 or voxel_bytes < 0:
            raise ValueError("nonnegative cache limits required")
        self.reader, self.raw_bytes, self.voxel_bytes = reader, raw_bytes, voxel_bytes
        self.roi_reader = roi_reader
        if not 0 <= time_tolerance_us < 50000:
            raise ValueError("invalid approximate voxel clock tolerance")
        self.approximate_voxels = approximate_voxels
        self.time_tolerance_us = time_tolerance_us
        self.raw: Raw | None = None
        self.span: tuple[int, int] | None = None
        self.identity: tuple[str, float] | None = None
        self.voxels: OrderedDict[tuple, Tensor] = OrderedDict()
        self.retained_voxel_bytes = 0
        self.last_diagnostics: dict[str, int | float] = {}

    def reset(self) -> None:
        """Invalidate raw data and exact voxel keys together."""
        self.raw, self.span, self.identity = None, None, None
        self.voxels.clear()
        self.retained_voxel_bytes = 0

    @staticmethod
    def _slice(raw: Raw, first: int, last: int) -> Raw:
        lo, hi = np.searchsorted(raw["t"], [first, last])
        return {key: value[lo:hi] for key, value in raw.items()}

    def _read(self, first: int, last: int) -> Raw:
        if self.raw is not None and self.span is not None:
            a, b = self.span
            if a <= first and last <= b:
                return self._slice(self.raw, first, last)
            if a <= first < b < last:
                tail = self.reader(b, last)
                prefix = self._slice(self.raw, first, b)
                raw = {k: np.concatenate((prefix[k], tail[k])) for k in prefix}
            else:
                raw = self.reader(first, last)
        else:
            raw = self.reader(first, last)
        if sum(x.nbytes for x in raw.values()) <= self.raw_bytes:
            self.raw, self.span = raw, (first, last)
        else:
            self.raw, self.span = None, None
        return raw

    def prepare(self, query: Query, anchors: list[int]) -> Tensor:
        """Encode only missing observations; exact duplicate windows share one tensor."""
        if not anchors or any(a > query.anchor for a in anchors):
            raise ValueError("nonempty causal observation list required")
        begin = time.perf_counter()
        read_ms = map_ms = encode_ms = 0.0
        if self.identity != (query.sequence, query.offset):
            self.reset()
            self.identity = (query.sequence, query.offset)
        windows = [query.at(a) for a in anchors]
        unique = {window for group in windows for window in group}
        keys = {w: (query.sequence, query.offset, query.roi, *w) for w in unique}
        missing = [w for w in unique if keys[w] not in self.voxels]
        local = {w: self.voxels[keys[w]] for w in unique if keys[w] in self.voxels}
        warped_count = 0
        if self.approximate_voxels:
            from .state import roi_distance

            for window in list(missing):
                for key in reversed(self.voxels):
                    seq, offset, roi, start, end = key
                    if seq != query.sequence or offset != query.offset:
                        continue
                    if not (
                        0 <= window[0] - start <= self.time_tolerance_us
                        and 0 <= window[1] - end <= self.time_tolerance_us
                    ):
                        continue
                    iou, scale = roi_distance(roi, query.roi)
                    if iou < 0.5 or scale > 0.35:
                        continue
                    local[window] = warp_voxel(self.voxels[key], roi, query.roi)
                    missing.remove(window)
                    warped_count += 1
                    break
        lookup_ms = (time.perf_counter() - begin) * 1000
        if missing:
            tick = time.perf_counter()
            first, last = min(w[0] for w in missing), max(w[1] for w in missing)
            raw = (
                self.roi_reader(first, last, query.roi, query.offset)
                if self.roi_reader is not None
                else self._read(first, last)
            )
            read_ms = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            x0, y0, x1, y1 = query.roi
            keep = (
                (raw["x"] >= max(0, x0) - query.offset)
                & (raw["x"] < min(1280, x1) - query.offset)
                & (raw["y"] >= max(0, y0))
                & (raw["y"] < min(720, y1))
            )
            cropped = {key: value[keep] for key, value in raw.items()}
            mapped = map_roi(cropped, query.roi, 128, query.offset)
            map_ms = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            for window in sorted(missing):
                value = encode_mapped(mapped, *window, 128, query.sequence)
                local[window] = value
                cost = value.numel() * value.element_size()
                if cost <= self.voxel_bytes:
                    while self.voxels and self.retained_voxel_bytes + cost > self.voxel_bytes:
                        _, removed = self.voxels.popitem(last=False)
                        self.retained_voxel_bytes -= removed.numel() * removed.element_size()
                    self.voxels[keys[window]] = value
                    self.retained_voxel_bytes += cost
            encode_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        result = torch.stack([torch.stack([local[w] for w in group]) for group in windows])
        stack_ms = (time.perf_counter() - tick) * 1000
        self.last_diagnostics = {
            "prep_lookup_warp_ms": lookup_ms,
            "prep_read_ms": read_ms,
            "prep_crop_map_ms": map_ms,
            "prep_encode_cache_ms": encode_ms,
            "prep_stack_ms": stack_ms,
            "observations_encoded": len(anchors),
            "unique_windows": len(unique),
            "voxel_misses": len(missing),
            "voxel_hits": len(unique) - len(missing),
            "approximate_voxel_hits": warped_count,
            "retained_voxel_bytes": self.retained_voxel_bytes,
            "retained_raw_bytes": sum(x.nbytes for x in self.raw.values()) if self.raw else 0,
        }
        return result
