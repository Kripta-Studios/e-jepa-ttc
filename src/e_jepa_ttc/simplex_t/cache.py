"""Deduplicated cache gathering, train-only normalization and registered controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.evaluation.stage63_65 import BUCKETS

from .controls import history_control


@dataclass(frozen=True)
class Normalizer:
    """A shared D-pool fit on unique consumed TRAIN observations, never queries twice."""

    mean: np.ndarray
    scale: np.ndarray
    consumed_ids_sha256: str


def fit_normalizer(
    features: np.ndarray, consumed: np.ndarray, train_allowed: np.ndarray
) -> Normalizer:
    """Fit in chunks so a memmap never becomes a duplicated context tensor."""
    ids = np.unique(consumed[consumed >= 0])
    if ids.size == 0 or ids.max() >= len(features) or not train_allowed[ids].all():
        raise ValueError("normalizer includes missing or non-TRAIN observations")
    if features.ndim != 2 or features.shape[1] not in {17, 145}:
        raise ValueError("invalid feature schema")
    count = 0
    mean = np.zeros(features.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(ids), 4096):
        block = np.asarray(features[ids[start : start + 4096]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite train features")
        delta = block.mean(0) - mean
        total = count + len(block)
        m2 += ((block - block.mean(0)) ** 2).sum(0) + delta**2 * count * len(block) / total
        mean += delta * len(block) / total
        count = total
    scale = np.sqrt(np.maximum(m2 / count, 0))
    scale[scale < 1e-8] = 1.0
    return Normalizer(mean, scale, hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest())


def training_mass(target_ttc: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Present-bucket-renormalized sequence mass for TRAIN only."""
    if target_ttc.ndim != 1 or sequences.shape != target_ttc.shape or not len(target_ttc):
        raise ValueError("target/sequence shape mismatch")
    if not np.isfinite(target_ttc).all():
        raise ValueError("nonfinite target")
    mass = np.zeros(len(target_ttc), dtype=np.float64)
    names = np.unique(sequences)
    for sequence in names:
        in_sequence = sequences == sequence
        present = []
        for _, low, high, weight in BUCKETS:
            mask = in_sequence & (target_ttc > low) & (target_ttc <= high)
            if mask.any():
                present.append((mask, weight))
        total_weight = sum(weight for _, weight in present)
        if not total_weight:
            raise ValueError("no valid training buckets")
        for mask, weight in present:
            mass[mask] = weight / (total_weight * len(names) * mask.sum())
    if np.any(mass <= 0) or not np.isclose(mass.sum(), 1):
        raise ValueError("TRAIN target outside benchmark buckets")
    return mass


@dataclass
class CachedQueries:
    """Gather only one minibatch from immutable memmaps and separate history indices.

    Instances must be constructed only after role/producer/cache freeze checks.
    Identity is the caller's complete content manifest, including normalization.
    The class deliberately accepts no raw paths or event loader.
    """

    features: np.ndarray
    anchor_us: np.ndarray
    available_us: np.ndarray
    history: np.ndarray
    target_phase: np.ndarray
    mass: np.ndarray
    normalizer: Normalizer
    identity_sha256: str
    length: int = 8
    control: str = "NONE"
    zero_latent: bool = False

    def __post_init__(self) -> None:
        if self.length not in {1, 4, 8, 16} or self.history.ndim != 2:
            raise ValueError("invalid history length")
        if self.history.shape[1] < self.length or self.history.dtype.kind != "i":
            raise ValueError("history index schema mismatch")
        if self.anchor_us.dtype != np.int64 or self.available_us.dtype != np.int64:
            raise ValueError("int64 times required")
        if len(self.anchor_us) != len(self.features) or len(self.available_us) != len(
            self.features
        ):
            raise ValueError("observation alignment mismatch")
        if self.mass.shape != (len(self.history),) or self.target_phase.shape != self.mass.shape:
            raise ValueError("query supervision alignment mismatch")
        if (
            not np.isfinite(self.mass).all()
            or np.any(self.mass < 0)
            or not np.isclose(self.mass.sum(), 1)
        ):
            raise ValueError("global population mass must sum to one")
        if self.control not in {"NONE", "PAST_REVERSED", "REPEAT_CURRENT"}:
            raise ValueError("unregistered feature control")
        if self.zero_latent and self.features.shape[1] != 145:
            raise ValueError("LATENT_ZERO requires the 145-D architecture")

    @property
    def population(self) -> int:
        """Number of unique supervised queries, independent of history length."""
        return len(self.history)

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Normalize then perturb features; preserve current, time slots and mask."""
        ids = query_ids.numpy()
        index = self.history[ids, -self.length :]
        valid = index >= 0
        if (index < -1).any() or (index >= len(self.features)).any():
            raise ValueError("cache index out of bounds")
        if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("noncontiguous/current-missing context")
        safe = np.maximum(index, 0)
        current = safe[:, -1]
        experts = np.asarray(self.features[current, 8:11], dtype=np.float32)
        x = ((self.features[safe] - self.normalizer.mean) / self.normalizer.scale).astype(
            np.float32
        )
        x[~valid] = 0
        if self.zero_latent:
            x[:, :, 17:] = 0
        anchors, availability = self.anchor_us[safe], self.available_us[safe]
        timing = np.zeros((*index.shape, 4), dtype=np.float32)
        timing[:, :, 0] = (self.anchor_us[current, None] - anchors) / 1e6
        timing[:, :, 1] = (self.available_us[current, None] - availability) / 1e6
        timing[:, :, 3] = (availability - anchors) / 1e6
        timing[:, 1:, 2] = np.where(valid[:, :-1], np.diff(anchors, axis=1) / 1e6, 0)
        timing[~valid] = 0
        if (timing[:, :, :2][valid] < 0).any():
            raise ValueError("future source dependency")
        x_tensor, mask = torch.from_numpy(x), torch.from_numpy(valid)
        if self.control != "NONE":
            x_tensor = history_control(
                x_tensor, mask, "shuffle" if self.control == "PAST_REVERSED" else "repeat"
            )
        return (
            x_tensor,
            torch.from_numpy(timing),
            mask,
            torch.from_numpy(experts),
            torch.from_numpy(self.target_phase[ids].astype(np.float32)),
            torch.from_numpy(self.mass[ids].astype(np.float32)),
        )
