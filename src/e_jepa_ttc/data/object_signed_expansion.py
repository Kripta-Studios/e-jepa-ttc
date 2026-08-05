"""Typed batches for geometry-conditioned signed-expansion TTC v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from e_jepa_ttc.data.garlttc_object_lhr import GarlTTCObjectLHRDataset
from e_jepa_ttc.models.object_signed_expansion import OBSERVABLE_MOTION_DIM


@dataclass
class ObjectSignedExpansionBatch:
    """Model inputs and supervision with an explicit no-leakage boundary."""

    events: torch.Tensor
    delta_t_s: torch.Tensor
    observable_motion: torch.Tensor
    jepa_context_motion: torch.Tensor
    precontext_motion_valid: torch.Tensor
    visible_heights_px: torch.Tensor
    target_ttc_s: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> ObjectSignedExpansionBatch:
        return ObjectSignedExpansionBatch(
            events=self.events.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            delta_t_s=self.delta_t_s.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            observable_motion=self.observable_motion.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            jepa_context_motion=self.jepa_context_motion.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            precontext_motion_valid=self.precontext_motion_valid.to(
                device=device,
                dtype=torch.bool,
                non_blocking=non_blocking,
            ),
            visible_heights_px=self.visible_heights_px.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            target_ttc_s=self.target_ttc_s.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            sequence_ids=self.sequence_ids,
            sample_tokens=self.sample_tokens,
            track_ids=self.track_ids,
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return only observable model inputs; labels cannot cross this API."""

        return {
            "events": self.events,
            "delta_t_s": self.delta_t_s,
            "observable_motion": self.observable_motion,
            "jepa_context_motion": self.jepa_context_motion,
            "precontext_motion_valid": self.precontext_motion_valid,
        }


class GarlTTCObjectSignedExpansionDataset(Dataset[dict[str, Any]]):
    """Official object cache requiring event and causal motion fields."""

    def __init__(
        self,
        manifest_path: str,
        *,
        splits: tuple[str, ...],
        event_field: str = "jepa_event_roi",
    ) -> None:
        self._dataset = GarlTTCObjectLHRDataset(
            manifest_path,
            splits=splits,
            event_field=event_field,
        )
        self.manifest = self._dataset.manifest
        self.event_field = event_field
        fields = set(self.manifest.get("model_input_fields", []))
        required = {
            event_field,
            "garl_delta_t_s",
            "observable_motion",
            "jepa_context_motion",
            "jepa_pair_valid",
            "precontext_motion_valid",
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                "Cache predates the causal geometry contract; missing "
                f"model inputs: {missing}"
            )
        forbidden = set(self.manifest.get("forbidden_model_input_fields", []))
        if {"ttc_s", "geometry_v2_target", "box3d_Fcam", "box3d_h"} - forbidden:
            raise ValueError("Cache manifest does not fail closed on privileged inputs")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


def _event_tensor(record: dict[str, Any], field: str) -> torch.Tensor:
    value = torch.as_tensor(record[field], dtype=torch.float32)
    if value.ndim != 4 or value.shape[0] != 2:
        raise ValueError(f"{field} must have shape [2,C,H,W]")
    if value.shape[-1] != value.shape[-2]:
        raise ValueError(f"{field} ROI must be square")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains non-finite values")
    return value


def _motion_tensor(record: dict[str, Any], field: str) -> torch.Tensor:
    value = torch.as_tensor(record[field], dtype=torch.float32)
    if value.shape != (OBSERVABLE_MOTION_DIM,):
        raise ValueError(
            f"{field} must have shape [{OBSERVABLE_MOTION_DIM}], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains non-finite values")
    return value


def collate_object_signed_expansion(
    records: list[dict[str, Any]],
    *,
    event_field: str = "jepa_event_roi",
) -> ObjectSignedExpansionBatch:
    """Collate model inputs and official supervision without field ambiguity."""

    if not records:
        raise ValueError("Signed-expansion collate received an empty batch")
    events = torch.stack([_event_tensor(record, event_field) for record in records])
    delta_t = torch.tensor(
        [float(record["garl_delta_t_s"]) for record in records],
        dtype=torch.float32,
    )
    observable = torch.stack(
        [_motion_tensor(record, "observable_motion") for record in records]
    )
    context = torch.stack(
        [_motion_tensor(record, "jepa_context_motion") for record in records]
    )
    pair_valid = torch.tensor(
        [bool(record["jepa_pair_valid"]) for record in records],
        dtype=torch.bool,
    )
    if not bool(pair_valid.all()):
        raise ValueError("All object endpoint pairs must be causally valid")
    context_valid = torch.tensor(
        [bool(record["precontext_motion_valid"]) for record in records],
        dtype=torch.bool,
    )
    heights = torch.stack(
        [
            torch.as_tensor(
                record["garl_visible_heights_px"],
                dtype=torch.float32,
            )
            for record in records
        ]
    )
    targets = torch.tensor(
        [float(record["ttc_s"]) for record in records],
        dtype=torch.float32,
    )
    if delta_t.shape != targets.shape or bool((delta_t <= 0).any()):
        raise ValueError("delta_t_s must be positive and aligned with TTC")
    if heights.shape != (len(records), 2):
        raise ValueError("Visible heights must collate to [B,2]")
    if not torch.isfinite(heights).all() or bool((heights <= 0).any()):
        raise ValueError("Visible heights must be finite and positive")
    if not torch.isfinite(targets).all() or bool((targets == 0).any()):
        raise ValueError("Official TTC targets must be finite and non-zero")
    if bool((delta_t / targets >= 1.0).any()):
        raise ValueError("Official TTC lies outside the LHR domain")
    return ObjectSignedExpansionBatch(
        events=events,
        delta_t_s=delta_t,
        observable_motion=observable,
        jepa_context_motion=context,
        precontext_motion_valid=context_valid,
        visible_heights_px=heights,
        target_ttc_s=targets,
        sequence_ids=[str(record["sequence_id"]) for record in records],
        sample_tokens=[str(record["sample_token"]) for record in records],
        track_ids=[str(record["track_id"]) for record in records],
    )


__all__ = [
    "GarlTTCObjectSignedExpansionDataset",
    "ObjectSignedExpansionBatch",
    "collate_object_signed_expansion",
]
