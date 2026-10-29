"""Optional multi-object query stage after the single-target gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.target_query import _mask_box


@dataclass
class MultiObjectQueryOutput:
    """Multiple object tokens, soft masks, boxes and no-object logits."""

    object_tokens: torch.Tensor
    mask_logits: torch.Tensor
    boxes_xyxy: torch.Tensor
    no_object_logits: torch.Tensor


class MultiObjectQueries(nn.Module):
    """Four-query extension; intentionally separate from the first candidate."""

    def __init__(self, dim: int, *, query_count: int = 4) -> None:
        super().__init__()
        if query_count <= 1:
            raise ValueError("MultiObjectQueries requires at least two queries.")
        self.queries = nn.Parameter(torch.empty(query_count, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.no_object = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> MultiObjectQueryOutput:
        """Decode query masks from ``[B,P,D]`` dense tokens."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,P,D].")
        height, width = spatial_shape
        logits = torch.einsum("bpd,qd->bqp", self.key(tokens), self.queries)
        logits = logits * tokens.shape[-1] ** -0.5
        weights = torch.softmax(logits, dim=-1)
        object_tokens = torch.einsum("bqp,bpd->bqd", weights, tokens)
        masks = torch.sigmoid(logits).reshape(tokens.shape[0], self.queries.shape[0], height, width)
        boxes = torch.stack([_mask_box(masks[:, index]) for index in range(masks.shape[1])], dim=1)
        return MultiObjectQueryOutput(
            object_tokens=object_tokens,
            mask_logits=logits.reshape(tokens.shape[0], self.queries.shape[0], height, width),
            boxes_xyxy=boxes,
            no_object_logits=self.no_object(object_tokens).squeeze(-1),
        )


__all__ = ["MultiObjectQueries", "MultiObjectQueryOutput"]
