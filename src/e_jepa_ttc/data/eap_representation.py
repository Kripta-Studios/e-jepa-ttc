"""Shared voxelization and representation utilities for eAP data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid, robust_normalize
from e_jepa_ttc.training.carla_jepa import (
    EVTTC_BASE_EVENT_CHANNELS,
    EVTTC_BASE_INPUT_CHANNELS,
)


def downsample_full_frame(
    events: dict[str, np.ndarray],
    *,
    sequence_id: str,
    start_us: int,
    end_us: int,
    width: int,
    height: int,
) -> EventBatch:
    source_width, source_height = EAP_IMAGE_SIZE
    x = np.minimum(
        np.asarray(events["x"], dtype=np.int64) * width // source_width,
        width - 1,
    ).astype(np.int32)
    y = np.minimum(
        np.asarray(events["y"], dtype=np.int64) * height // source_height,
        height - 1,
    ).astype(np.int32)
    return EventBatch(
        x=x,
        y=y,
        t_us=np.asarray(events["t"], dtype=np.int64),
        polarity=np.where(np.asarray(events["p"]) > 0, 1, -1).astype(np.int8),
        width=width,
        height=height,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )


def base_compatible_voxel(events: EventBatch, *, bins: int) -> torch.Tensor:
    voxel = encode_voxel_grid(events, bins=bins, normalize=True)
    tensor = np.zeros(
        (EVTTC_BASE_INPUT_CHANNELS, events.height, events.width),
        dtype=np.float32,
    )
    tensor[:EVTTC_BASE_EVENT_CHANNELS] = voxel
    duration_s = max((events.t_end_us - events.t_start_us) * 1e-6, 1e-6)
    tensor[10] = np.log1p(events.num_events)
    tensor[11] = np.log1p(events.num_events / duration_s)
    return torch.from_numpy(tensor)


def event_voxel_with_scalars(events: EventBatch, *, bins_per_polarity: int) -> torch.Tensor:
    """Encode an arbitrary temporal resolution plus count and rate channels.

    Five bins per polarity reproduce the first twelve channels of
    :func:`base_compatible_voxel`. Higher resolutions do not allocate the nine
    historical compatibility channels.
    """

    if bins_per_polarity <= 0:
        raise ValueError("bins_per_polarity must be positive")
    voxel = encode_voxel_grid(events, bins=bins_per_polarity, normalize=True)
    event_channels = 2 * bins_per_polarity
    if voxel.shape[0] != event_channels:
        raise RuntimeError("voxel channel count differs from the temporal schema")
    tensor = np.zeros(
        (event_channels + 2, events.height, events.width),
        dtype=np.float32,
    )
    tensor[:event_channels] = voxel
    duration_s = max((events.t_end_us - events.t_start_us) * 1e-6, 1e-6)
    tensor[event_channels] = np.log1p(events.num_events)
    tensor[event_channels + 1] = np.log1p(events.num_events / duration_s)
    return torch.from_numpy(tensor)


def base_compatible_voxel_chunks(
    chunks: Iterable[dict[str, np.ndarray]],
    *,
    sequence_id: str,
    start_us: int,
    end_us: int,
    width: int,
    height: int,
    bins: int,
) -> torch.Tensor:
    """Voxelize an iterable of raw event chunks with bounded temporary memory."""

    return base_compatible_voxel_windows_chunks(
        chunks,
        windows=((start_us, end_us),),
        sequence_id=sequence_id,
        width=width,
        height=height,
        bins=bins,
    )[0]


def base_compatible_voxel_windows_chunks(
    chunks: Iterable[dict[str, np.ndarray]],
    *,
    windows: Sequence[tuple[int, int]],
    sequence_id: str,
    width: int,
    height: int,
    bins: int,
) -> torch.Tensor:
    """Voxelize several disjoint or overlapping windows from one chunk stream.

    The raw event stream is read once over the enclosing interval while each
    requested window gets an independent accumulator and event count. This
    preserves per-window normalization and avoids four separate HDF5 reads for
    a JEPA context/future tuple.
    """

    if not windows:
        raise ValueError("At least one event window is required.")
    if any(end_us <= start_us for start_us, end_us in windows):
        raise ValueError("All event windows require positive duration.")
    raw = [np.zeros((EVTTC_BASE_EVENT_CHANNELS, height, width), dtype=np.float32) for _ in windows]
    event_counts = [0] * len(windows)
    for chunk in chunks:
        timestamps = np.asarray(chunk["t"], dtype=np.int64)
        for index, (start_us, end_us) in enumerate(windows):
            selected = (timestamps >= start_us) & (timestamps < end_us)
            if not np.any(selected):
                continue
            selected_events = {key: np.asarray(values)[selected] for key, values in chunk.items()}
            event_batch = downsample_full_frame(
                selected_events,
                sequence_id=sequence_id,
                start_us=start_us,
                end_us=end_us,
                width=width,
                height=height,
            )
            raw[index] += encode_voxel_grid(event_batch, bins=bins, normalize=False)
            event_counts[index] += event_batch.num_events

    tensors: list[torch.Tensor] = []
    for (start_us, end_us), values, event_count in zip(
        windows,
        raw,
        event_counts,
        strict=True,
    ):
        tensor = np.zeros((EVTTC_BASE_INPUT_CHANNELS, height, width), dtype=np.float32)
        tensor[:EVTTC_BASE_EVENT_CHANNELS] = robust_normalize(values)
        duration_s = max((end_us - start_us) * 1e-6, 1e-6)
        tensor[10] = np.log1p(event_count)
        tensor[11] = np.log1p(event_count / duration_s)
        tensors.append(torch.from_numpy(tensor))
    return torch.stack(tensors)
