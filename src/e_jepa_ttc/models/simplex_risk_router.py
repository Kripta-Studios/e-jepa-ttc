"""Reference kernels for ordinal, convex, 1-Lipschitz absolute-phase risk.

This is a new research hypothesis, not a reproduction of a published TTC model.
Predictions remain one of the three frozen expert TTCs; no TTC averaging occurs.
Integration must add the repository's authorization, provenance and resume layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def benchmark_phase(ttc: np.ndarray) -> np.ndarray:
    values = np.asarray(ttc, dtype=np.float64)
    if not np.isfinite(values).all() or not np.all((values < 0) | (values > 0.1)):
        raise ValueError("TTC must be finite and lie in (-inf,0) union (0.1,inf)")
    return -np.log1p(-0.1 / values)


def costs_from_logits(phases: Tensor, logits: Tensor, *, constrained: bool = True) -> Tensor:
    """Return three relative predicted MiD costs /100, in A5,C2F,PAIR order.

    A conditional expectation of |a-Y| is convex and 1-Lipschitz in a.
    Its two secant slopes at three sorted expert phases therefore obey
    -1 <= s0 <= s1 <= 1. Softmax increments enforce those constraints.
    They are *not* calibrated probabilities of TTC bins.
    """
    if phases.ndim != 2 or phases.shape[1] != 3 or logits.shape != (len(phases), 2):
        raise ValueError("expected phases [N,3] and logits [N,2]")
    if not torch.isfinite(phases).all() or not torch.isfinite(logits).all():
        raise ValueError("phase/logit inputs must be finite")
    if phases.dtype != logits.dtype or phases.device != logits.device:
        raise ValueError("phase and logit dtype/device must match")
    order = torch.argsort(phases, dim=1, stable=True)
    sorted_phase = phases.gather(1, order)
    gaps = sorted_phase[:, 1:] - sorted_phase[:, :-1]
    if constrained:
        p = torch.softmax(torch.cat((logits, torch.zeros_like(logits[:, :1])), dim=1), dim=1)
        slopes = torch.stack((2.0 * p[:, 0] - 1.0, 1.0 - 2.0 * p[:, 2]), dim=1)
    else:
        # Exactly the same zero-head initial slopes as the constrained arm.
        slopes = logits + logits.new_tensor([-1.0 / 3.0, 1.0 / 3.0])
    increments = 100.0 * gaps * slopes  # 10000 MiD / target scaling 100
    sorted_cost = torch.cat((torch.zeros_like(increments[:, :1]), increments.cumsum(1)), dim=1)
    expert_cost = torch.zeros_like(sorted_cost).scatter(1, order, sorted_cost)
    return expert_cost - expert_cost[:, :1]


class SimplexRiskRouter(nn.Module):
    """Matched 17- or 45-input network; controls mask features after normalization."""

    def __init__(self, input_dim: int = 17, constrained: bool = True) -> None:
        super().__init__()
        if input_dim not in (17, 45):
            raise ValueError("this protocol only authorizes 17 or 45 input features")
        self.input_dim = input_dim
        self.constrained = constrained
        self.body = nn.Sequential(nn.Linear(input_dim, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.head = nn.Linear(16, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, normalized_features: Tensor, phases: Tensor) -> Tensor:
        if normalized_features.ndim != 2 or normalized_features.shape != (
            len(phases),
            self.input_dim,
        ):
            raise ValueError("feature shape differs from frozen protocol")
        if not torch.isfinite(normalized_features).all():
            raise ValueError("features must be finite")
        return costs_from_logits(
            phases, self.head(self.body(normalized_features)), constrained=self.constrained
        )


def relative_cost_targets(target_phase: Tensor, expert_phases: Tensor) -> Tensor:
    if target_phase.shape != (len(expert_phases),) or expert_phases.shape != (len(target_phase), 3):
        raise ValueError("target and expert shapes disagree")
    if not torch.isfinite(target_phase).all() or not torch.isfinite(expert_phases).all():
        raise ValueError("targets and phases must be finite")
    costs = 100.0 * (expert_phases - target_phase[:, None]).abs()
    return costs - costs[:, :1]


def weighted_cost_loss(
    predicted: Tensor, target: Tensor, global_mass: Tensor, train_rows: int
) -> Tensor:
    """Unbiased uniform-minibatch estimate of the full macro-weighted MSE.

    global_mass contains the sampled entries of the *global* train mass (sum=1
    over the complete train table). Never renormalize within this minibatch.
    """
    if predicted.ndim != 2 or predicted.shape[1] != 3 or target.shape != predicted.shape:
        raise ValueError("relative cost shapes disagree")
    if global_mass.shape != (len(predicted),) or train_rows <= 0:
        raise ValueError("mass/train size mismatch")
    if not all(torch.isfinite(v).all() for v in (predicted, target, global_mass)):
        raise ValueError("loss inputs must be finite")
    if (global_mass < 0).any():
        raise ValueError("negative mass is forbidden")
    # A5 is identically zero; do not dilute by including it in the output mean.
    return (train_rows * global_mass * (predicted[:, 1:] - target[:, 1:]).square().mean(1)).mean()


def hard_select(costs: Tensor, expert_ttc: Tensor) -> tuple[Tensor, Tensor]:
    if costs.ndim != 2 or costs.shape[1] != 3 or expert_ttc.shape != costs.shape:
        raise ValueError("cost/TTC shapes disagree")
    if not torch.isfinite(costs).all() or not torch.isfinite(expert_ttc).all():
        raise ValueError("nonfinite expert or cost")
    if not ((expert_ttc <= -0.1) | (expert_ttc > 0.1)).all():
        raise ValueError("expert TTC violates the strict scientific prediction domain")
    chosen = costs.argmin(1)  # fixed A5 -> C2F -> PAIR tie order
    return chosen, expert_ttc.gather(1, chosen[:, None]).squeeze(1)
