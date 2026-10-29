from __future__ import annotations

import torch

from e_jepa_ttc.models.ttc_readout import SignedTTCReadout


def test_signed_ttc_readout_is_finite_and_differentiable() -> None:
    torch.manual_seed(53)
    readout = SignedTTCReadout(8)
    embedding = torch.randn(4, 8, requires_grad=True)
    prediction = readout(embedding)
    prediction.sum().backward()
    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()
    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()
