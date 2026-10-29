"""Task-specific routing across backbone depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass
class AttentionResidualOutput:
    """Task tokens and auditable layer weights."""

    task_tokens: dict[str, torch.Tensor]
    task_weights: dict[str, torch.Tensor]


class RMSNorm(nn.Module):
    """Minimal RMS normalization used before routing."""

    def __init__(self, dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = value.square().mean(dim=-1, keepdim=True).add(self.epsilon).sqrt()
        return value / rms * self.scale


class TaskSpecificAttentionResiduals(nn.Module):
    """Learn a separate normalized layer mixture for every downstream task."""

    def __init__(
        self,
        dim: int,
        layer_count: int,
        *,
        tasks: tuple[str, ...] = ("mask", "motion", "geometry", "risk"),
    ) -> None:
        super().__init__()
        if layer_count < 2 or not tasks:
            raise ValueError("AttnRes needs at least two layers and one task.")
        self.tasks = tasks
        self.layer_count = layer_count
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(layer_count))
        self.queries = nn.ParameterDict({task: nn.Parameter(torch.empty(dim)) for task in tasks})
        for query in self.queries.values():
            nn.init.zeros_(query)

    def forward(self, layer_tokens: tuple[torch.Tensor, ...]) -> AttentionResidualOutput:
        """Route aligned ``[B,T,P,D]`` layer tokens per task."""

        if len(layer_tokens) != self.layer_count:
            raise ValueError("layer_tokens count differs from configured layer_count.")
        reference_shape = layer_tokens[0].shape
        if any(tokens.shape != reference_shape for tokens in layer_tokens):
            raise ValueError("All layer token tensors must have the same shape.")
        values = torch.stack(layer_tokens, dim=-2)
        normalized_keys = torch.stack(
            [norm(tokens) for norm, tokens in zip(self.norms, layer_tokens, strict=True)],
            dim=-2,
        )
        outputs: dict[str, torch.Tensor] = {}
        weights_by_task: dict[str, torch.Tensor] = {}
        scale = reference_shape[-1] ** -0.5
        for task, query in self.queries.items():
            logits = torch.einsum("btpld,d->btpl", normalized_keys, query) * scale
            weights = functional.softmax(logits, dim=-1)
            outputs[task] = (values * weights[..., None]).sum(dim=-2)
            weights_by_task[task] = weights
        return AttentionResidualOutput(task_tokens=outputs, task_weights=weights_by_task)


__all__ = ["AttentionResidualOutput", "RMSNorm", "TaskSpecificAttentionResiduals"]
