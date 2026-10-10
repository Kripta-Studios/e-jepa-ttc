"""Exact producer collation without materializing discarded RGB fields.

The prepared-cache reader and DINO reader retain responsibility for shard and
row validation. The canonical training loop still validates the resulting
ObjectEventV4Batch. This adapter changes only which dataset fields are stacked;
it leaves dataset access order, DINO access, dtype conversion and metadata intact.
"""

# ruff: noqa: ANN401 -- adapter accepts the frozen source and test doubles
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

PRODUCER_TENSOR_FIELDS = (
    "rgb",
    "delta_t_s",
    "boxes_in_crop_xyxy",
    "visible_heights_px",
    "roi_xyxy",
    "target_ttc",
)


def producer_batch(source: Any, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
    """Build the canonical producer batch, skipping tensors it never consumes."""
    if modality != "rgb" or not indices:
        raise ValueError("RGBProducerSource only serves nonempty RGB batches")
    samples = [source.dataset[int(index)] for index in indices]
    counts = {len(sample["rgb"]) for sample in samples}
    if len(counts) != 1:
        raise ValueError("T2 and T3 observations require separate sampler buckets")
    # Identical as_tensor -> stack -> float order to RGBProducerSource.batch.
    collated = {
        field: torch.stack([torch.as_tensor(sample[field]) for sample in samples])
        for field in PRODUCER_TENSOR_FIELDS
    }
    metadata = {
        field: [sample[field] for sample in samples]
        for field in ("sample_token", "track_id", "sequence_id")
    }
    values, valid = source._dino_for_indices(indices)
    return ObjectEventV4Batch(
        events=collated["rgb"].float(),
        delta_t_s=collated["delta_t_s"].float(),
        observable_motion=torch.empty((len(indices), 0), dtype=torch.float32),
        visible_heights_px=collated["visible_heights_px"].float(),
        target_ttc_s=collated["target_ttc"].float(),
        boxes_xyxy=collated["boxes_in_crop_xyxy"].float(),
        common_square_xyxy=collated["roi_xyxy"].float(),
        sequence_ids=metadata["sequence_id"],
        sample_tokens=metadata["sample_token"],
        track_ids=metadata["track_id"],
        dinov3_relation_targets=torch.from_numpy(values.astype(np.float32, copy=False)),
        dinov3_relation_valid=torch.from_numpy(valid.astype(np.bool_, copy=False)),
    )


@contextmanager
def installed_collate(source: Any) -> Iterator[None]:
    """Install only on this source; restore inherited or overridden methods exactly.

    An outer runtime must retain the admitted input-I/O retry wrapper, because an
    instance method takes precedence over a patched class method.
    """
    had_override = "batch" in vars(source)
    previous = vars(source).get("batch")
    source.batch = MethodType(producer_batch, source)
    try:
        yield
    finally:
        if had_override:
            source.batch = previous
        else:
            del source.batch
