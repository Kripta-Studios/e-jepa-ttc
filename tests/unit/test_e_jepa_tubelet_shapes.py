from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig


def _model() -> EJEPATubeletLHR:
    return EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
        )
    ).eval()


def test_tubelet_keeps_batch_time_patch_and_embedding_axes() -> None:
    model = _model()
    with torch.inference_mode():
        features = model.forward_features(torch.randn(2, 3, 4, 17, 19))

    assert features.tokens.shape == (2, 3, 4, 32)
    assert features.valid_patch_mask.shape == (2, 3, 4)
    assert features.geometry.padded_height == 24
    assert features.geometry.padded_width == 24


def test_tubelet_rejects_an_ambiguous_flattened_layout() -> None:
    with pytest.raises(ValueError, match="shape"):
        _model().forward_features(torch.randn(2, 3, 4, 17))


def test_ttc_readout_uses_queries_by_default_and_keeps_mean_as_control() -> None:
    model = _model()
    assert model.config.pooling == "query"
    assert model.query_tokens.shape == (8, 32)
    control = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
            pooling="mean",
        )
    ).eval()
    assert control.config.pooling == "mean"
    assert control.query_tokens is None
