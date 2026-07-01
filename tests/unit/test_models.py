import torch

from e_jepa_ttc.models import (
    EventTokenTransformerEncoder,
    EventTokenTransformerRegressor,
    TinyCNNEncoder,
    TinyCNNRegressor,
)


def test_tiny_cnn_encoder_and_regressor_shapes() -> None:
    x = torch.randn(2, 6, 24, 32)
    encoder = TinyCNNEncoder(in_channels=6, width=16)
    regressor = TinyCNNRegressor(in_channels=6, width=16)

    encoded = encoder(x)
    tokens = encoder.forward_tokens(x)
    pred = regressor(x)

    assert encoded.shape == (2, 64)
    assert tokens.shape == (2, 12, 64)
    assert pred.shape == (2,)


def test_event_token_transformer_shapes() -> None:
    x = torch.randn(2, 6, 24, 32)
    encoder = EventTokenTransformerEncoder(
        in_channels=6,
        embed_dim=48,
        patch_size=8,
        depth=2,
        num_heads=4,
    )
    regressor = EventTokenTransformerRegressor(
        in_channels=6,
        embed_dim=48,
        patch_size=8,
        depth=2,
        num_heads=4,
    )

    encoded = encoder(x)
    tokens = encoder.forward_tokens(x)
    pred = regressor(x)

    assert encoded.shape == (2, 48)
    assert tokens.shape == (2, 12, 48)
    assert pred.shape == (2,)
