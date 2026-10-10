"""Causal sensor histories for the explicitly exposed Dev32 transfer protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.evttc_rgb_transfer.inputs import _rgb_box
from operational.evttc_transfer.inputs import EvTTCEventReader, encode_union

EVENT_LAGS_US = np.arange(3, -1, -1, dtype=np.int64) * 100_000


def prepare_event(row: Mapping[str, Any]) -> dict[str, Any]:
    """Generate four native-cadence observations, not historical 50ms H8 inputs."""
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    if windows.shape != (3, 2) or not np.array_equal(windows[:, 1] - windows[:, 0], [100_000] * 3):
        raise ValueError("Three 100ms event windows are required")
    if not np.array_equal(np.diff(windows[:, 1]), [100_000] * 2):
        raise ValueError("Event endpoints must follow native 100ms cadence")
    if windows[-1, 1] != int(row["anchor_us"]):
        raise ValueError("Event reference must be the last context endpoint")
    path = Path(row["raw_path"])
    stat = path.stat()
    if row["raw_stat"] != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        # Manifests may retain additional diagnostic stat fields.
        if any(
            int(row["raw_stat"][key]) != value
            for key, value in (("size_bytes", stat.st_size), ("mtime_ns", stat.st_mtime_ns))
        ):
            raise ValueError("Raw EvTTC source changed after query selection")
    if int(row["anchor_us"]) - int(windows[0, 0] - EVENT_LAGS_US[0]) > 650_000:
        raise ValueError("Event history exceeds its complete-input 650ms span")
    with EvTTCEventReader(path) as reader:
        # The retained encoder has an H16 storage ABI. Only the last four real
        # observations are valid; unused slots never reach a producer or head.
        storage_lags = np.arange(15, -1, -1, dtype=np.int64) * 100_000
        storage_valid = np.arange(16) >= 12
        own = encode_union(
            reader,  # pyright: ignore[reportArgumentType] -- shared read_window reader contract
            windows,
            storage_lags,
            storage_valid,
            tuple(float(value) for value in row["square_xyxy"]),
            sequence_id=str(row["sequence_id"]),
            roi_size=128,
            event_pixel_diff=0.0,
            retained_bytes_max=1024 * 1024**2,
        )[-4:]
    return {"own_events": own.numpy().astype(np.float32, copy=False)}


def sensor_triplets(clock: np.ndarray, query: int) -> list[np.ndarray]:
    """Select actual frames at native 100ms cadence (within 1ms), never synthesize."""
    if clock.ndim != 1 or (len(clock) > 1 and np.any(np.diff(clock) <= 0)):
        raise ValueError("RGB sensor clock must be strictly increasing")
    available = np.flatnonzero(clock <= query)
    if len(available) < 2:
        return []
    chain = [int(available[-1])]
    while len(chain) < 10:
        target = int(clock[chain[-1]]) - 100_000
        insertion = int(np.searchsorted(clock, target))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < chain[-1]]
        if not candidates:
            break
        previous = min(candidates, key=lambda index: (abs(int(clock[index]) - target), index))
        if abs(int(clock[previous]) - target) > 1_000 or query - int(clock[previous]) > 650_000:
            break
        chain.append(previous)
    chain.reverse()
    observations = [np.asarray(chain[end - 2 : end + 1]) for end in range(2, len(chain))]
    if len(chain) == 2 and int(clock[chain[0]]) - 100_000 < int(clock[0]):
        observations = [np.asarray(chain)]
    return observations[-8:]


def raw_rgb_observations(
    row: Mapping[str, Any],
) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], str]:
    """Decode real triplets in a causal query ROI, declaring its delayed availability."""
    query = int(row["anchor_us"])
    sources = list(row["bbox_sources"])[-2:]
    # These public boxes are part of the already-frozen label-free query manifest.
    boxes = [_rgb_box(source) for source in sources]
    square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
    result = []
    with h5py.File(Path(row["raw_path"]), "r") as handle:
        data, timestamps = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
        if not isinstance(data, h5py.Dataset) or not isinstance(timestamps, h5py.Dataset):
            return [], "RGB_DATASET_MISSING"
        clock = np.asarray(timestamps, dtype=np.int64)
        groups = sensor_triplets(clock, query)
        if not groups:
            return [], "INSUFFICIENT_CAUSAL_RGB_HISTORY"
        frames = {
            int(index): crop_uint8(np.asarray(data[int(index)], np.uint8), square)
            for index in np.unique(np.concatenate(groups))
        }
        for ids in groups:
            sensor = clock[ids]
            rgb = np.ascontiguousarray(
                np.stack([frames[int(index)] for index in ids]).astype(np.float32) / 255
            )
            result.append(
                (
                    rgb,
                    {
                        "first_sensor_ts_us": int(sensor[0]),
                        "anchor_us": int(sensor[-1]),
                        "available_us": query,
                        "timestamps_us": sensor.tolist(),
                        "delta_t_s": (np.diff(sensor) / 1e6).tolist(),
                        "square_xyxy": list(square),
                    },
                )
            )
    return result, ""
