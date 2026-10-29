"""Adapter exposing signed V8 cache rows to the established A5 collate contract."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from e_jepa_ttc.data.scientific_recovery_v8_cache import ScientificRecoveryV8CacheDataset


class V8ToObjectEventV4Dataset(Dataset[dict[str, Any]]):
    """Map V8 cache rows into training-only A5 batch fields without model leakage."""

    def __init__(
        self,
        cache: ScientificRecoveryV8CacheDataset,
        *,
        outer_fold: int,
        split: str,
        steps: int = 3,
    ) -> None:
        if split not in {"train", "dev"}:
            raise ValueError("split must be train or dev")
        if steps not in {2, 3}:
            raise ValueError("V8 causal-scale adapter supports only 2 or 3 steps")
        self.cache = cache
        self.steps = int(steps)
        self.indices = [
            index
            for index in range(len(cache))
            if (int(cache[index]["outer_fold"]) != outer_fold) == (split == "train")
        ]
        if not self.indices:
            raise ValueError(f"V8 {split} fold view is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.cache[self.indices[index]]
        values = np.asarray(row["representation"], dtype=np.float32)
        if values.ndim != 4 or values.shape[0] < self.steps:
            raise ValueError("V8 representation does not contain the requested endpoint count")
        values = values[-self.steps :].copy()
        boxes = np.asarray(row["endpoint_boxes_xyxy"], dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] < self.steps or boxes.shape[1] != 4:
            raise ValueError("V8 endpoint boxes do not match requested endpoint count")
        boxes = boxes[-self.steps :].copy()
        square = np.asarray(row["common_roi_xyxy"], dtype=np.float32)
        width = float(square[2] - square[0])
        height = float(square[3] - square[1])
        if width <= 0 or height <= 0:
            raise ValueError("V8 common ROI must have positive area")
        scale_x = values.shape[-1] / width
        scale_y = values.shape[-2] / height
        adapted = boxes.copy()
        adapted[:, (0, 2)] = (adapted[:, (0, 2)] - square[0]) * scale_x
        adapted[:, (1, 3)] = (adapted[:, (1, 3)] - square[1]) * scale_y
        delta = (int(row["endpoint_us"][-1]) - int(row["endpoint_us"][-2])) / 1_000_000.0
        if delta <= 0.0 or float(row["target_ttc"]) == 0.0:
            raise ValueError("V8 adapter requires positive delta and nonzero TTC")
        return {
            "event_v4_common_roi": values,
            "_event_v4_expected_steps": self.steps,
            "_event_v4_expected_channels": int(values.shape[1]),
            "garl_delta_t_s": np.float32(delta),
            "observable_motion": np.zeros(18, dtype=np.float32),
            "garl_visible_heights_px": np.asarray(row["visible_heights_px"], dtype=np.float32),
            "ttc_s": np.float32(row["target_ttc"]),
            "event_v4_boxes_xyxy": adapted,
            "event_v4_common_square_xyxy": square,
            "sequence_id": row["sequence_id"],
            "sample_token": row["sample_token"],
            "track_id": row["track_id"],
            "sample_weight": np.float32(row["sample_weight"]),
        }

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(range(len(self))),)
