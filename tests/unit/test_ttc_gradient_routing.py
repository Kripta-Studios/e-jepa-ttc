from __future__ import annotations

import torch

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig


def test_ttc_loss_reaches_the_patch_encoder_and_online_path() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
        )
    )
    output = model(torch.randn(2, 2, 4, 16, 16))
    loss = output.ttc_mean_seconds.mean()
    loss.backward()
    assert model.patch_embed.weight.grad is not None
    assert model.ttc_head[-1].weight.grad is not None
    assert torch.isfinite(model.patch_embed.weight.grad).all()
