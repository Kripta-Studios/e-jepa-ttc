import torch

from e_jepa_ttc.models import (
    EventTokenTransformerEncoder,
    EventTokenTransformerRegressor,
    EventTubeletTransformerEncoder,
    EventTubeletTransformerRegressor,
    TinyCNNEncoder,
    TinyCNNRegressor,
    build_encoder,
    build_regressor,
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
    intermediate = encoder.forward_intermediate_tokens(x, (0, 1))
    pred = regressor(x)

    assert encoded.shape == (2, 48)
    assert tokens.shape == (2, 12, 48)
    assert [layer.shape for layer in intermediate] == [(2, 12, 48), (2, 12, 48)]
    assert pred.shape == (2,)


def test_large_token_transformer_factory_shapes() -> None:
    x = torch.randn(2, 6, 24, 32)
    encoder = build_encoder("token-transformer-large", in_channels=6)
    regressor = build_regressor("token-transformer-large", in_channels=6)

    encoded = encoder(x)
    tokens = encoder.forward_tokens(x)
    intermediate = encoder.forward_intermediate_tokens(x, (2, 5))
    pred = regressor(x)

    assert encoded.shape == (2, 256)
    assert tokens.shape == (2, 12, 256)
    assert [layer.shape for layer in intermediate] == [(2, 12, 256), (2, 12, 256)]
    assert pred.shape == (2,)


def test_event_tubelet_transformer_shapes_with_aux_channels() -> None:
    x = torch.randn(2, 12, 32, 48)
    encoder = EventTubeletTransformerEncoder(
        in_channels=12,
        embed_dim=48,
        event_bins=5,
        patch_size=16,
        temporal_patch_size=1,
        depth=2,
        num_heads=4,
    )
    regressor = EventTubeletTransformerRegressor(
        in_channels=12,
        embed_dim=48,
        event_bins=5,
        patch_size=16,
        temporal_patch_size=1,
        depth=2,
        num_heads=4,
    )

    encoded = encoder(x)
    tokens = encoder.forward_tokens(x)
    intermediate = encoder.forward_intermediate_tokens(x, (0, 1))
    pred = regressor(x)

    assert encoded.shape == (2, 48)
    assert tokens.shape == (2, 30, 48)
    assert [layer.shape for layer in intermediate] == [(2, 30, 48), (2, 30, 48)]
    assert pred.shape == (2,)


def test_event_tubelet_transformer_factory_shapes() -> None:
    x = torch.randn(2, 21, 32, 48)
    encoder = build_encoder("event-tubelet-transformer", in_channels=21)
    regressor = build_regressor("event-tubelet-transformer", in_channels=21)

    encoded = encoder(x)
    tokens = encoder.forward_tokens(x)
    intermediate = encoder.forward_intermediate_tokens(x, (1, 5))
    pred = regressor(x)

    assert encoded.shape == (2, 192)
    assert tokens.shape == (2, 30, 192)
    assert [layer.shape for layer in intermediate] == [(2, 30, 192), (2, 30, 192)]
    assert pred.shape == (2,)
