"""Strict input-only observation boundary; local format adapters must prove lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .history import Observation, build_history

INPUT_FIELDS = (
    "token",
    "sequence_id",
    "acquisition_group",
    "track_id",
    "producer_family_sha256",
    "label_anchor_us",
    "latest_sensor_dependency_us",
    "latest_roi_dependency_us",
    "raw_binding_hash",
    "role",
)


@dataclass(frozen=True)
class InputObservation:
    """Audited dependencies are separate from TTC numeric supervision."""

    token: str
    sequence_id: str
    acquisition_group: str
    track_id: str
    producer_family_sha256: str
    label_anchor_us: int
    latest_sensor_dependency_us: int
    latest_roi_dependency_us: int
    raw_binding_hash: str
    role: str

    def __post_init__(self) -> None:
        for name in ("label_anchor_us", "latest_sensor_dependency_us", "latest_roi_dependency_us"):
            value = getattr(self, name)
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("integer int64 microseconds required")
        if self.role not in {"TRAIN", "OLD_DEV"}:
            raise ValueError("protected/confirmation/unknown role")
        for name in ("producer_family_sha256", "raw_binding_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("lowercase SHA256 identity required")
        if not all((self.token, self.sequence_id, self.acquisition_group, self.track_id)):
            raise ValueError("complete observation identity required")

    @property
    def available_us(self) -> int:
        """Last sensor or ROI dependency, including permitted current-input delay."""
        return max(self.latest_sensor_dependency_us, self.latest_roi_dependency_us)

    @classmethod
    def project_trusted_record(cls, raw: dict[str, Any]) -> InputObservation:
        """Discard labels immediately after an explicitly authorized legacy read.

        This whitelist does not establish source eligibility. The caller must
        independently prove that raw membership was not filtered by target.
        """
        return cls(**{key: raw[key] for key in INPUT_FIELDS})


def indices(
    rows: list[InputObservation],
    queries: np.ndarray,
    *,
    length: int,
    allowed_groups: set[str],
    producer_fit_ancestors: dict[str, set[str]],
    outer_heldout: set[str],
) -> np.ndarray:
    """Validate all consumed families before constructing chronological history."""
    simple = []
    for row in rows:
        if row.acquisition_group not in allowed_groups:
            raise ValueError("source group not in permitted role universe")
        if row.producer_family_sha256 not in producer_fit_ancestors:
            raise ValueError("missing transitive producer ancestors")
        fitted = producer_fit_ancestors[row.producer_family_sha256]
        if fitted & ({row.acquisition_group} | outer_heldout):
            raise ValueError("ancestor includes observation or outer-held-out group")
        simple.append(
            Observation(
                row.token,
                row.sequence_id,
                row.track_id,
                row.producer_family_sha256,
                row.label_anchor_us,
                row.available_us,
                row.role,
            )
        )
    return build_history(simple, queries, length=length)


def validate_model_fields(batch: dict[str, Any]) -> None:
    """Reject targets and identity metadata at the neural input boundary."""
    expected = {"features", "timing", "valid", "current_expert_phase"}
    if set(batch) != expected:
        raise ValueError("model batch accepts only features/timing/mask/current expert phases")
