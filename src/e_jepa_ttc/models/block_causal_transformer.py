"""Block-causal attention: full spatial access, causal temporal access."""

from __future__ import annotations

import torch
from torch import nn


def block_causal_attention_mask(
    steps: int,
    patches: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a bool mask where queries cannot attend to later frames."""

    if steps <= 0 or patches <= 0:
        raise ValueError("steps and patches must be positive.")
    # ``expand`` + ``reshape`` has stable ONNX lowering, unlike
    # ``repeat_interleave`` in the PyTorch 2.11 exporter.
    time_index = torch.arange(steps, device=device).unsqueeze(1).expand(steps, patches).reshape(-1)
    return time_index[None, :] > time_index[:, None]


class BlockCausalTransformer(nn.Module):
    """Transformer encoder preserving Patch Policy temporal causality."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
        maximum_tokens: int = 8192,
    ) -> None:
        super().__init__()
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive.")
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.empty(maximum_tokens, dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix ``[B,T,P,D]`` with a frame-level causal mask."""

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,P,D].")
        batch, steps, patches, dim = tokens.shape
        flat = tokens.reshape(batch, steps * patches, dim)
        if flat.shape[1] > self.position.shape[0]:
            raise ValueError(
                f"Token count {flat.shape[1]} exceeds configured maximum {self.position.shape[0]}."
            )
        flat = flat + self.position[: flat.shape[1]]
        mask = block_causal_attention_mask(steps, patches, device=tokens.device)
        # The mask is block-causal, not the standard triangular causal mask:
        # patches from the same instant remain mutually visible.  Passing
        # ``is_causal=False`` prevents PyTorch from trying to infer standard
        # causality by converting a symbolic tensor to ``bool`` during ONNX
        # export while still applying our explicit attention mask.
        return self.encoder(flat, mask=mask, is_causal=False).reshape(batch, steps, patches, dim)


__all__ = ["BlockCausalTransformer", "block_causal_attention_mask"]
