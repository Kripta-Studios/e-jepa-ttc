"""Unquantized native40-plane Garl inputs with bounded read-only sensor access."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_square_box,
    official_timevolume_roi_np,
)
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool


def read_native_window(
    pool: ReaderPool, path: Path, start: int, end: int, offset: int = 5, cap: int = 256 * 1024**2
) -> dict:
    """Preserve release floor-ms boundaries, sensor clipping and int16 arithmetic."""
    reader = pool.get(path)
    index = reader.datasets["ms_to_idx"]
    if end > int(reader.datasets["events/t"][-1]):
        raise ValueError("native window exceeds stream support")
    lo, hi = int(index[start // 1000]), int(index[math.floor(end / 1000)])
    if (hi - lo) * 12 > cap:
        raise InterruptedError("native full-sensor window exceeds256MiB bounded extraction")
    # Same operation order as the source: offset before int16 conversion.
    x = (reader.datasets["events/x"][lo:hi] + offset).astype(np.int16)
    y = reader.datasets["events/y"][lo:hi].astype(np.int16)
    t = reader.datasets["events/t"][lo:hi].astype(np.int64)
    width, height = EAP_IMAGE_SIZE
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return {"x": x[keep], "y": y[keep], "t": t[keep]}


def native_feature(raw: dict, square: tuple, size: int = 128) -> torch.Tensor:
    """Keep the source's global first-event time origin and FP32 resize semantics."""
    feature, _ = official_timevolume_roi_np(
        square, raw["x"].astype(np.int64), raw["y"].astype(np.int64), raw["t"], number_of_planes=20
    )
    return official_resize_feature(feature, (size, size))


def encode_record(
    row: dict, pool: ReaderPool, raw_train_root: Path, fy: float = 1694.1323524131867
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Native two endpoints and training-only visible-height supervision."""
    path = raw_train_root / row["sequence_id"] / "events.h5"
    if len(row["event_windows_us"]) != 2 or len(row["boxes_xyxy"]) != 2:
        raise ValueError("native40-channel endpoint count is not two")
    boxes = [
        cast(tuple[float, float, float, float], tuple(map(float, b))) for b in row["boxes_xyxy"]
    ]
    tensors = []
    visible = []
    for i in range(2):
        square = official_square_box(boxes, i)
        start, end = map(int, row["event_windows_us"][i])
        raw = read_native_window(pool, path, start, end)
        if len(raw["t"]) == 0:
            raise ValueError("empty native source endpoint; pre-update admission required")
        tensors.append(native_feature(raw, square))
        corners = np.asarray(row["box3d_Fcam"][i], np.float64)
        depth = float(corners[:, 2].min())
        if depth <= 0:
            raise ValueError("invalid native visible-height geometry")
        visible.append(fy * float(row["box3d_h"]) / depth * (128 / (square[3] - square[1])))
    return (
        torch.cat(tensors),
        torch.tensor(visible, dtype=torch.float32),
        float(row["frame_ttc"][-1]),
    )


def inference_record(row: dict, pool: ReaderPool, raw_train_root: Path) -> torch.Tensor:
    """Native sensor/ROI-only forward input; no TRAIN or OLD_DEV label access."""
    boxes = [
        cast(tuple[float, float, float, float], tuple(map(float, b))) for b in row["boxes_xyxy"]
    ]
    if len(boxes) != 2 or len(row["event_windows_us"]) != 2:
        raise ValueError("native inference requires exactly two endpoints")
    path = raw_train_root / row["sequence_id"] / "events.h5"
    tensors = []
    for i, (start, end) in enumerate(row["event_windows_us"]):
        raw = read_native_window(pool, path, int(start), int(end))
        if not len(raw["t"]):
            raise ValueError("native inference endpoint is empty; no query dropping allowed")
        tensors.append(native_feature(raw, official_square_box(boxes, i)))
    return torch.cat(tensors)
