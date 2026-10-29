"""Canonical encoder constructors and a common dense/global output view."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models import build_encoder
from e_jepa_ttc.models.e_jepa_tubelet_lhr import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
)
from e_jepa_ttc.models.tiny_cnn import TinyCNNEncoder
from e_jepa_ttc.models.token_transformer import (
    EventTokenTransformerEncoder,
    EventTubeletTransformerEncoder,
)


@dataclass(frozen=True)
class EncoderOutput:
    """Global embedding and dense tokens returned by adapter encoders."""

    global_embedding: torch.Tensor
    dense_tokens: torch.Tensor
    feature_maps: tuple[torch.Tensor, ...] = ()


def encode_with_dense_output(encoder: nn.Module, inputs: torch.Tensor) -> EncoderOutput:
    """Adapt supported encoders to the shared ``[B,D]``/``[B,N,D]`` contract."""

    forward_tokens = getattr(encoder, "forward_tokens", None)
    if callable(forward_tokens):
        tokens = forward_tokens(inputs)
    else:
        tokens = encoder(inputs)
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("Encoder must return a tensor or provide forward_tokens().")
    if tokens.ndim == 2:
        dense = tokens[:, None, :]
    elif tokens.ndim == 3:
        dense = tokens
    else:
        raise ValueError(f"Expected [B,D] or [B,N,D] encoder output, got {tokens.shape}.")
    return EncoderOutput(global_embedding=dense.mean(dim=1), dense_tokens=dense)


__all__ = [
    "EJEPATubeletLHR",
    "EJEPATubeletLHRConfig",
    "EncoderOutput",
    "EventTokenTransformerEncoder",
    "EventTubeletTransformerEncoder",
    "TinyCNNEncoder",
    "build_encoder",
    "encode_with_dense_output",
]
