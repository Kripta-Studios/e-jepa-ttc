"""Small, auditable router over physical TTC experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.attention_residual_router import RMSNorm


@dataclass
class GeometryRouterOutput:
    """Mixture value, weights and stability diagnostics."""

    inverse_ttc: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor
    balance_loss: torch.Tensor


class StableGeometryRouter(nn.Module):
    """RMS-normalized soft router with a permanently active shared path."""

    def __init__(
        self,
        feature_dim: int,
        expert_count: int,
        *,
        hidden_dim: int = 64,
        shared_weight_floor: float = 0.10,
        inference_top_k: int | None = None,
    ) -> None:
        super().__init__()
        if expert_count < 2 or not 0.0 <= shared_weight_floor < 1.0:
            raise ValueError("Router needs >=2 experts and a valid shared-weight floor.")
        if inference_top_k is not None and not 1 <= inference_top_k <= expert_count:
            raise ValueError("inference_top_k must be within the expert count.")
        self.expert_count = expert_count
        self.shared_weight_floor = shared_weight_floor
        self.inference_top_k = inference_top_k
        self.norm = RMSNorm(feature_dim)
        self.router = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, expert_count),
        )

    def forward(
        self,
        features: torch.Tensor,
        estimates: torch.Tensor,
        confidence: torch.Tensor,
    ) -> GeometryRouterOutput:
        """Route using physical/quality features, never sequence identifiers."""

        if estimates.shape != confidence.shape or estimates.shape[-1] != self.expert_count:
            raise ValueError("Estimate/confidence shapes must end in expert_count.")
        logits = self.router(self.norm(features))
        logits = logits + confidence.clamp_min(1e-6).log()
        weights = functional.softmax(logits, dim=-1)
        floor = self.shared_weight_floor
        shared = torch.full_like(weights, floor / self.expert_count)
        weights = weights * (1.0 - floor) + shared
        if not self.training and self.inference_top_k is not None:
            _, indices = torch.topk(weights, self.inference_top_k, dim=-1)
            active = torch.zeros_like(weights).scatter_(-1, indices, 1.0)
            weights = weights * active
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        valid = estimates.isfinite() & (estimates >= 0)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        safe_estimates = torch.where(valid, estimates, torch.zeros_like(estimates))
        inverse_ttc = (weights * safe_estimates).sum(dim=-1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        mean_usage = weights.reshape(-1, self.expert_count).mean(dim=0)
        balance_loss = (mean_usage - 1.0 / self.expert_count).square().mean()
        return GeometryRouterOutput(
            inverse_ttc=inverse_ttc,
            weights=weights,
            entropy=entropy,
            balance_loss=balance_loss,
        )


__all__ = ["GeometryRouterOutput", "StableGeometryRouter"]
