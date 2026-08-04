"""Typed object-centric batches for the official GarlTTC LHR cache.

The cache contains both supervision-only geometry and model inputs. This module
creates an explicit boundary: only ROI event tensors and the measured temporal
interval are returned by :meth:`ObjectLHRBatch.model_inputs`. TTC, visible
heights, and masks remain supervision and can never be passed accidentally to
the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch.utils.data import Dataset


@dataclass
class ObjectLHRBatch:
    """One collated object-centric LHR batch."""

    events: torch.Tensor
    delta_t_s: torch.Tensor
    visible_heights_px: torch.Tensor
    target_ttc_s: torch.Tensor
    masks: torch.Tensor
    mask_valid: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]

    def to(self, device: torch.device, *, non_blocking: bool = True) -> ObjectLHRBatch:
        """Move tensor members while preserving string provenance."""

        return ObjectLHRBatch(
            events=self.events.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            delta_t_s=self.delta_t_s.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            visible_heights_px=self.visible_heights_px.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            target_ttc_s=self.target_ttc_s.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            masks=self.masks.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            mask_valid=self.mask_valid.to(
                device=device,
                dtype=torch.bool,
                non_blocking=non_blocking,
            ),
            sequence_ids=self.sequence_ids,
            sample_tokens=self.sample_tokens,
            track_ids=self.track_ids,
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return the complete and only model-input mapping."""

        return {"events": self.events, "delta_t_s": self.delta_t_s}


class GarlTTCObjectLHRDataset(Dataset[dict[str, Any]]):
    """Lazy official-cache view requiring the object-JEPA ROI extension.

    The cache module is imported only when a dataset is instantiated. This keeps
    lightweight geometry/loss unit tests independent from HDF5/plugin setup.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        splits: tuple[str, ...],
        event_field: str = "jepa_event_roi",
    ) -> None:
        from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset

        self._dataset = GarlTTCLHRCacheDataset(manifest_path, splits=splits)
        self.manifest = self._dataset.manifest
        self.event_field = event_field
        fields = set(self.manifest.get("model_input_fields", []))
        if event_field not in fields:
            raise ValueError(
                f"Cache does not expose required model input {event_field!r}; "
                "rebuild it with the object-LHR cache builder from this patch."
            )
        extension = self.manifest.get("object_lhr_extension")
        if not isinstance(extension, dict) or int(extension.get("version", 0)) < 1:
            raise ValueError("Cache is missing object_lhr_extension version 1 metadata.")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._dataset[index]
        if self.event_field not in record:
            raise KeyError(f"Cache record is missing {self.event_field!r}")
        return record


def _event_tensor(record: dict[str, Any], field: str) -> torch.Tensor:
    value = torch.as_tensor(record[field], dtype=torch.float32)
    if value.ndim != 4 or value.shape[0] != 2:
        raise ValueError(f"{field} must have shape [2,C,H,W], got {tuple(value.shape)}")
    if value.shape[-1] != value.shape[-2]:
        raise ValueError(f"{field} ROI must be square, got {tuple(value.shape[-2:])}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains non-finite values")
    return value


def collate_object_lhr(
    records: list[dict[str, Any]],
    *,
    event_field: str = "jepa_event_roi",
) -> ObjectLHRBatch:
    """Collate cache records and fail closed on malformed geometry."""

    if not records:
        raise ValueError("Object-LHR collate received an empty batch")
    events = torch.stack([_event_tensor(record, event_field) for record in records])
    delta_t = torch.tensor([float(record["garl_delta_t_s"]) for record in records])
    heights = torch.stack(
        [
            torch.as_tensor(record["garl_visible_heights_px"], dtype=torch.float32)
            for record in records
        ]
    )
    targets = torch.tensor([float(record["ttc_s"]) for record in records], dtype=torch.float32)
    if heights.shape != (len(records), 2):
        raise ValueError(
            "garl_visible_heights_px must collate to [B,2], "
            f"got {tuple(heights.shape)}"
        )
    if not torch.isfinite(heights).all() or bool((heights <= 0).any()):
        raise ValueError("Visible-height supervision must be finite and strictly positive")
    if not torch.isfinite(targets).all() or bool((targets == 0).any()):
        raise ValueError("Official TTC supervision must be finite and non-zero")
    ratio = 1.0 - delta_t / targets
    if bool((ratio <= 0).any()) or not torch.isfinite(ratio).all():
        raise ValueError("Official TTC implies a non-positive height ratio")

    roi_size = events.shape[-1]
    masks: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for record in records:
        if "garl_mask_pair" in record:
            mask = torch.as_tensor(record["garl_mask_pair"], dtype=torch.float32)
            valid = torch.as_tensor(record.get("garl_mask_valid", [True, True]), dtype=torch.bool)
        else:
            mask = torch.zeros((2, 1, roi_size, roi_size), dtype=torch.float32)
            valid = torch.zeros(2, dtype=torch.bool)
        if mask.shape != (2, 1, roi_size, roi_size):
            raise ValueError(f"garl_mask_pair has invalid shape {tuple(mask.shape)}")
        if valid.shape != (2,):
            raise ValueError(f"garl_mask_valid has invalid shape {tuple(valid.shape)}")
        masks.append(mask.clamp(0.0, 1.0))
        valids.append(valid)

    return ObjectLHRBatch(
        events=events,
        delta_t_s=delta_t,
        visible_heights_px=heights,
        target_ttc_s=targets,
        masks=torch.stack(masks),
        mask_valid=torch.stack(valids),
        sequence_ids=[str(record["sequence_id"]) for record in records],
        sample_tokens=[str(record["sample_token"]) for record in records],
        track_ids=[str(record["track_id"]) for record in records],
    )


__all__ = [
    "GarlTTCObjectLHRDataset",
    "ObjectLHRBatch",
    "collate_object_lhr",
]
