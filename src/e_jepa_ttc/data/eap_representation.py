"""Shared voxelization and representation utilities for eAP data."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
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
