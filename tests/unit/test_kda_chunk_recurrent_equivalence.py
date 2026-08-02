from __future__ import annotations

import torch

from e_jepa_ttc.models.temporal_kda import (
    KDALayoutMetadata,
    KimiDeltaAttention,
    kimi_delta_recurrence,
    kimi_delta_recurrence_chunked,
)


def _recurrence_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(43)
    query = torch.randn(2, 9, 3, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn(2, 9, 3, 2, 5)
    retention = torch.sigmoid(torch.randn_like(query))
    beta = torch.sigmoid(torch.randn(2, 9, 3, 2))
    return query, key, value, retention, beta


def test_chunked_recurrence_matches_one_pass_and_keeps_fp32_state() -> None:
    query, key, value, retention, beta = _recurrence_inputs()
    full, _full_state = kimi_delta_recurrence(
        query,
        key,
        value,
        retention,
        beta,
        return_state=True,
    )
    chunked, state = kimi_delta_recurrence_chunked(
        query,
        key,
        value,
        retention,
        beta,
        chunk_size=4,
    )
    assert torch.allclose(full, chunked, atol=1e-6, rtol=1e-6)
    assert state.dtype == torch.float32


def test_kda_layer_chunk_state_matches_full_forward_and_reset_is_explicit() -> None:
    torch.manual_seed(47)
    layer = KimiDeltaAttention(16, heads=4).eval()
    values = torch.randn(1, 8, 3, 16)
    metadata = KDALayoutMetadata(1, 8, 3, 16)
    chunk_metadata = KDALayoutMetadata(1, 3, 3, 16)
    tail_metadata = KDALayoutMetadata(1, 2, 3, 16)
    with torch.inference_mode():
        full = layer(values, metadata=metadata)
        first, state = layer.forward_chunk(values[:, :3], metadata=chunk_metadata)
        second, state = layer.forward_chunk(values[:, 3:6], state, metadata=chunk_metadata)
        third, state = layer.forward_chunk(values[:, 6:], state, metadata=tail_metadata)
        reset, _reset_state = layer.forward_chunk(values[:, :3], metadata=chunk_metadata)
    assert torch.allclose(full, torch.cat((first, second, third), dim=1), atol=1e-5, rtol=1e-5)
    assert state.recurrent.dtype == torch.float32
    assert torch.allclose(reset, first, atol=1e-6, rtol=1e-6)
