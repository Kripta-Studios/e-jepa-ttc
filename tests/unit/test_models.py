import torch

from e_jepa_ttc.models import TinyCNNEncoder, TinyCNNRegressor


def test_tiny_cnn_encoder_and_regressor_shapes() -> None:
    x = torch.randn(2, 6, 24, 32)
    encoder = TinyCNNEncoder(in_channels=6, width=16)
    regressor = TinyCNNRegressor(in_channels=6, width=16)

    encoded = encoder(x)
    pred = regressor(x)

    assert encoded.shape == (2, 64)
    assert pred.shape == (2,)
