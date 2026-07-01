"""Model definitions."""

from torch import nn

from e_jepa_ttc.models.tiny_cnn import TinyCNNEncoder, TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import (
    EventTokenTransformerEncoder,
    EventTokenTransformerRegressor,
    EventTubeletTransformerEncoder,
    EventTubeletTransformerRegressor,
)

MODEL_NAMES = (
    "tiny-cnn",
    "token-transformer",
    "token-transformer-large",
    "event-tubelet-transformer",
    "event-tubelet-transformer-large",
)


def build_encoder(name: str, *, in_channels: int) -> nn.Module:
    """Build an encoder by public model name."""

    if name == "tiny-cnn":
        return TinyCNNEncoder(in_channels=in_channels)
    if name == "token-transformer":
        return EventTokenTransformerEncoder(in_channels=in_channels)
    if name == "token-transformer-large":
        return EventTokenTransformerEncoder(
            in_channels=in_channels,
            embed_dim=256,
            depth=6,
            num_heads=8,
        )
    if name == "event-tubelet-transformer":
        return EventTubeletTransformerEncoder(in_channels=in_channels)
    if name == "event-tubelet-transformer-large":
        return EventTubeletTransformerEncoder(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
        )
    msg = f"Unknown model {name!r}; expected one of {MODEL_NAMES}."
    raise ValueError(msg)


def build_regressor(name: str, *, in_channels: int) -> nn.Module:
    """Build a TTC regressor by public model name."""

    if name == "tiny-cnn":
        return TinyCNNRegressor(in_channels=in_channels)
    if name == "token-transformer":
        return EventTokenTransformerRegressor(in_channels=in_channels)
    if name == "token-transformer-large":
        return EventTokenTransformerRegressor(
            in_channels=in_channels,
            embed_dim=256,
            depth=6,
            num_heads=8,
        )
    if name == "event-tubelet-transformer":
        return EventTubeletTransformerRegressor(in_channels=in_channels)
    if name == "event-tubelet-transformer-large":
        return EventTubeletTransformerRegressor(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
        )
    msg = f"Unknown model {name!r}; expected one of {MODEL_NAMES}."
    raise ValueError(msg)


__all__ = [
    "MODEL_NAMES",
    "EventTubeletTransformerEncoder",
    "EventTubeletTransformerRegressor",
    "EventTokenTransformerEncoder",
    "EventTokenTransformerRegressor",
    "TinyCNNEncoder",
    "TinyCNNRegressor",
    "build_encoder",
    "build_regressor",
]
