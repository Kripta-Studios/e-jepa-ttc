"""P2 RGB adapter that reuses admitted real frames and their own sensor clock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.rgb_port.data import RGBPortDataset
from operational.range_transfer.common import read, sha

from .common import owner
from .data import role_rows


class RGBPairDataset:
    """Slice distinct native RGB pairs before collating mixed original T2/T3 rows."""

    def __init__(self, role: str) -> None:
        if role not in {"P", "H", "V"}:
            raise ValueError("Unknown RGB role")
        manifest = owner() / f"{role}_MANIFEST.json"
        if sha(Path(read(manifest)["rows_path"])) != read(manifest)["rows_sha256"]:
            raise ValueError("RGB metadata changed")
        self.source = RGBPortDataset(manifest, role=role)
        self.rows = role_rows(role)
        self.population = len(self.rows)
        self.role = role
        # This admitted cohort aligns labels with the latest annotation frame.
        # A delayed-query deployment needs a separately registered clock adapter.
        if not (self.rows.query_delay_us == 0).all():
            raise ValueError("RGB cohort requires an explicit delayed-query recipe")
        if self.source.prepared_cache is None:
            raise ValueError(
                "Reuse the admitted decoded RGB cache before creating a new data route"
            )

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        """Keep label/geometry tensors separate from RGB and measured frame separation."""
        if not indices:
            raise ValueError("Nonempty indices required")
        samples = [self.source[index] for index in indices]
        deltas = []
        for index, sample in zip(indices, samples, strict=True):
            if sample["sample_token"] != str(self.rows.iloc[index].sample_token):
                raise ValueError("RGB query identity changed")
            times = np.asarray(sample["sensor_frame_times_us"])[-2:]
            annotations = np.asarray(sample["annotation_frame_times_us"])[-2:]
            if len(times) != 2 or times[1] <= times[0] or annotations[1] <= annotations[0]:
                raise ValueError("Two distinct chronological real frames required")
            if int(annotations[-1]) != int(sample["rgb_anchor_us"]):
                raise ValueError("RGB annotation anchor changed")
            delta = np.float32((int(times[1]) - int(times[0])) / 1_000_000)
            native_delta = np.asarray(sample["delta_t_s"]).reshape(-1)[-1]
            if delta != native_delta:
                raise ValueError("RGB sensor delta differs from its native pair")
            pixels = np.asarray(sample["rgb"])[-2:]
            if (
                pixels.shape != (2, 3, 128, 128)
                or pixels.dtype != np.float32
                or not np.isfinite(pixels).all()
                or (pixels < 0).any()
                or (pixels > 1).any()
            ):
                raise ValueError("RGB must retain native FP32 [0,1] pixels")
            deltas.append(delta)

        def tensor(name: str, *, pair: bool = False) -> torch.Tensor:
            values = [
                np.asarray(sample[name])[-2:] if pair else np.asarray(sample[name])
                for sample in samples
            ]
            return torch.from_numpy(np.stack(values))

        return ObjectEventV4Batch(
            events=tensor("rgb", pair=True),
            delta_t_s=torch.tensor(deltas),
            observable_motion=torch.empty(len(indices), 0),
            visible_heights_px=tensor("visible_heights_px", pair=True),
            target_ttc_s=tensor("target_ttc"),
            boxes_xyxy=tensor("boxes_in_crop_xyxy", pair=True),
            common_square_xyxy=tensor("roi_xyxy").float(),
            sequence_ids=[sample["sequence_id"] for sample in samples],
            sample_tokens=[sample["sample_token"] for sample in samples],
            track_ids=[sample["track_id"] for sample in samples],
        )

    def close(self) -> None:
        """Release only this adapter's read-only handles and bounded cache."""
        if self.source.reader is not None:
            self.source.reader.close()
        if self.source.prepared_cache is not None:
            self.source.prepared_cache._loaded.clear()
