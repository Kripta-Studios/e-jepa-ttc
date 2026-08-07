"""Training helpers for Object Event TTC v4.18 radial physics bottleneck."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_18 import MonotoneOddPhysicsHead


def train_monotone_head(
    features: torch.Tensor,
    target_expansion: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    if len(train_indices) == 0:
        raise ValueError("train_indices cannot be empty")
    torch.manual_seed(seed)
    model = MonotoneOddPhysicsHead(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x = features[train_indices].float()
    y = (target_expansion[train_indices] < 0.0).float()
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append({"epoch": float(epoch), "loss": value})
    return best_state, history


@torch.no_grad()
def sign_from_negative_logit(logit: torch.Tensor) -> torch.Tensor:
    return torch.where(logit >= 0.0, -torch.ones_like(logit), torch.ones_like(logit))


@torch.no_grad()
def raw_physics_score(features: torch.Tensor) -> torch.Tensor:
    """Positive score means approaching; robust median avoids one noisy proxy."""
    return features.median(dim=1).values


def sign_accuracy(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    neg = target < 0.0
    pos = target >= 0.0
    neg_acc = float(np.mean(prediction[neg] < 0.0)) if np.any(neg) else 1.0
    pos_acc = float(np.mean(prediction[pos] >= 0.0)) if np.any(pos) else 1.0
    return {
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_sign_accuracy": 0.5 * (pos_acc + neg_acc),
    }


__all__ = [
    "raw_physics_score",
    "sign_accuracy",
    "sign_from_negative_logit",
    "train_monotone_head",
]
