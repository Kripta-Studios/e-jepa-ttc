from __future__ import annotations

import torch

from e_jepa_ttc.models.block_causal_transformer import (
    BlockCausalTransformer,
    block_causal_attention_mask,
)


def test_block_causal_mask_is_frame_causal_but_spatially_bidirectional() -> None:
    mask = block_causal_attention_mask(steps=3, patches=4)
    assert not mask[0, 1]
    assert mask[0, 4]
    assert not mask[4, 0]
    assert not mask[4, 7]
    assert mask[4, 8]


def test_dense_block_causal_forward_preserves_tubelet_layout() -> None:
    model = BlockCausalTransformer(16, heads=4, depth=1).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 5, 16))
    assert output.shape == (2, 3, 5, 16)
