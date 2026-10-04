"""Native three-feature Garl contract on the unchanged two-layer GRU160 family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.simplex_t.cache import Normalizer
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import pinball


@dataclass(frozen=True)
class NativeHeadConfig:
    """Only the authorized3-feature projection; retain GRU160 and residual output."""

    feature_count: int = 3
    hidden: int = 160
    backbone: str = "gru"
    output_mode: str = "residual"

    def __post_init__(self) -> None:
        if (self.feature_count, self.hidden, self.backbone, self.output_mode) != (
            3,
            160,
            "gru",
            "residual",
        ):
            raise ValueError("only the fixed Garl-H1/H8 contract is authorized")


def model() -> TemporalRefiner:
    """Reuse every recurrent/head operation; no new encoder or temporal architecture."""
    return TemporalRefiner(cast(TemporalConfig, NativeHeadConfig()))


def objective(
    output: dict[str, Tensor],
    truth: Tensor,
    experts: Tensor,
    mass: Tensor,
    population: int,
    *,
    selector_only: bool = False,
) -> Tensor:
    """Historical point and quantile losses with exactly lambda_cost=0."""
    if selector_only or experts.shape != (len(truth), 3) or not torch.isfinite(mass).all():
        raise ValueError("invalid native residual head contract")
    point = (output["point_phase"] - truth).abs() / 0.03
    quantile = (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9)) / 0.03
    return (population * mass * (point + 0.1 * quantile)).mean()


@dataclass
class NativeHeadSource:
    """A compact native observation table, query history and TRAIN-only statistics."""

    features: np.ndarray
    times: np.ndarray
    history: np.ndarray
    truth: np.ndarray
    mass: np.ndarray
    normalizer: Normalizer
    identity_sha256: str
    length: int

    @property
    def population(self) -> int:
        """Number of unique queries; independent of cold-start history length."""
        return len(self.history)

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Keep phase-current anchor, four parent times and current mandatory."""
        ids = query_ids.numpy()
        indices = self.history[ids, -self.length :]
        valid = indices >= 0
        if (
            self.length not in (1, 8)
            or not valid[:, -1].all()
            or (valid[:, :-1] & ~valid[:, 1:]).any()
        ):
            raise ValueError("native histories must retain a contiguous valid suffix")
        safe = np.maximum(indices, 0)
        raw = self.features[safe]
        x = ((raw - self.normalizer.mean) / self.normalizer.scale).astype(np.float32)
        timing = self.times[ids, -self.length :].copy()
        if self.length == 1:
            timing[:, :, 2] = 0
        x[~valid], timing[~valid] = 0, 0
        experts = np.repeat(self.features[safe[:, -1], :1], 3, axis=1).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(timing),
            torch.from_numpy(valid),
            torch.from_numpy(experts),
            torch.from_numpy(self.truth[ids].astype(np.float32)),
            torch.from_numpy(self.mass[ids].astype(np.float32)),
        )


def normalize(features: np.ndarray, history: np.ndarray) -> Normalizer:
    """Three native columns; count each unique consumed TRAIN observation once."""
    import hashlib

    ids = np.unique(history[history >= 0])
    if features.shape[1] != 3 or not len(ids):
        raise ValueError("native three-column TRAIN source required")
    values = features[ids].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite native TRAIN inputs")
    mean, scale = values.mean(0), values.std(0)
    scale[scale < 1e-8] = 1
    return Normalizer(mean, scale, hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest())
