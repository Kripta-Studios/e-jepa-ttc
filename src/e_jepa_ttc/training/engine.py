"""Minimal device-aware training engine for tensor loss callables."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EpochResult:
    """Aggregated loss and batch count for one explicit epoch."""

    loss: float
    batches: int


class TrainingEngine:
    """Small reusable engine; dataset-specific trainers remain canonical modules."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)

    def run_epoch(
        self,
        loader: Iterable[Any],
        loss_fn: Callable[[nn.Module, Any], torch.Tensor],
        *,
        train: bool = True,
    ) -> EpochResult:
        """Run a caller-supplied loss function without inventing batch semantics."""

        self.model.train(train)
        values: list[float] = []
        for batch in loader:
            loss = loss_fn(self.model, batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Training engine received a non-finite scalar loss.")
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            values.append(float(loss.detach().cpu()))
        if not values:
            raise ValueError("Training engine received an empty loader.")
        return EpochResult(loss=float(sum(values) / len(values)), batches=len(values))


__all__ = ["EpochResult", "TrainingEngine"]
