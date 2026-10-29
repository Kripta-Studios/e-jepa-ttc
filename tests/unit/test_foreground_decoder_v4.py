from __future__ import annotations

import torch

from e_jepa_ttc.models.foreground_training_decoder import ForegroundTrainingDecoder


def test_compact_foreground_decoder_returns_training_only_logits() -> None:
    decoder = ForegroundTrainingDecoder(
        dim=8,
        hidden_dim=16,
        output_size=16,
        output_channels=4,
    )
    tokens = torch.randn(2, 9, 8, requires_grad=True)
    logits = decoder(tokens, (3, 3))
    logits.mean().backward()
    assert logits.shape == (2, 4, 16, 16)
    assert tokens.grad is not None
    assert torch.isfinite(logits).all()
