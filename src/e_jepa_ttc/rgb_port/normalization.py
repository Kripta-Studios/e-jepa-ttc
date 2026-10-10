"""Role-bound RGB feature normalization with unique-observation fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RGB_PHASE17_NAMES = (
    "rgb_luminance_mean",
    "rgb_spatial_gradient_log1p",
    "rgb_a5_flow",
    "rgb_a5_margin",
    "rgb_a5_log_variance",
    "rgb_c2f_flow",
    "rgb_c2f_margin",
    "rgb_c2f_log_variance",
    "rgb_a5_phase",
    "rgb_c2f_phase",
    "rgb_pair_phase",
    "rgb_pair_minus_a5",
    "rgb_pair_minus_c2f",
    "rgb_c2f_minus_a5",
    "rgb_abs_pair_minus_a5",
    "rgb_abs_pair_minus_c2f",
    "rgb_abs_c2f_minus_a5",
)
RGB_PHASE17_SHA256 = hashlib.sha256(
    json.dumps(RGB_PHASE17_NAMES, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True)
class FrozenNormalizer:
    """A modality- and role-specific mean/scale transform."""

    mean: np.ndarray
    scale: np.ndarray
    fit_role: str
    schema_sha256: str
    producer_sha256: str
    unique_ids_sha256: str
    modality: str = "rgb"

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.mean) or not np.isfinite(array).all():
            raise ValueError("normalizer input violates its frozen schema")
        return np.ascontiguousarray((array - self.mean) / self.scale, dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "rgb_port_normalizer_v1",
            "modality": self.modality,
            "fit_role": self.fit_role,
            "schema_sha256": self.schema_sha256,
            "producer_sha256": self.producer_sha256,
            "unique_ids_sha256": self.unique_ids_sha256,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FrozenNormalizer:
        """Load and validate a frozen normalizer receipt."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "rgb_port_normalizer_v1":
            raise ValueError("unsupported RGB-PORT normalizer schema")
        result = cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            fit_role=str(payload["fit_role"]),
            schema_sha256=str(payload["schema_sha256"]),
            producer_sha256=str(payload["producer_sha256"]),
            unique_ids_sha256=str(payload["unique_ids_sha256"]),
            modality=str(payload["modality"]),
        )
        if result.mean.ndim != 1 or result.scale.shape != result.mean.shape:
            raise ValueError("invalid frozen normalizer arrays")
        if (
            not np.isfinite(result.mean).all()
            or not np.isfinite(result.scale).all()
            or np.any(result.scale <= 0)
        ):
            raise ValueError("invalid frozen normalizer values")
        return result

    def validate_endpoint(
        self,
        *,
        modality: str,
        fit_role: str,
        schema_sha256: str,
        producer_sha256: str,
    ) -> None:
        """Reject reuse across modalities, roles, schemas or producer endpoints."""
        expected = (self.modality, self.fit_role, self.schema_sha256, self.producer_sha256)
        observed = (modality, fit_role, schema_sha256, producer_sha256)
        if observed != expected:
            raise ValueError("normalizer endpoint identity mismatch")


def fit_normalizer(
    values: np.ndarray,
    observation_ids: list[str],
    *,
    fit_role: str,
    expected_role: str,
    producer_sha256: str,
    schema_sha256: str = RGB_PHASE17_SHA256,
    modality: str = "rgb",
) -> FrozenNormalizer:
    """Fit once per unique observation and reject V or cross-role fitting."""
    array = np.asarray(values, dtype=np.float64)
    if fit_role == "V" or fit_role != expected_role:
        raise ValueError("normalization may only fit its preregistered P or H role")
    if modality not in {"event", "rgb"}:
        raise ValueError("normalizer modality must be event or rgb")
    if array.ndim != 2 or len(array) != len(observation_ids) or not np.isfinite(array).all():
        raise ValueError("normalizer values/IDs are invalid")
    first: dict[str, int] = {}
    for index, identity in enumerate(observation_ids):
        first.setdefault(str(identity), index)
    ids = sorted(first)
    unique = array[[first[identity] for identity in ids]]
    mean = unique.mean(axis=0)
    scale = unique.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return FrozenNormalizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        fit_role=fit_role,
        schema_sha256=schema_sha256,
        producer_sha256=producer_sha256,
        unique_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        modality=modality,
    )


__all__ = ["FrozenNormalizer", "RGB_PHASE17_NAMES", "RGB_PHASE17_SHA256", "fit_normalizer"]
