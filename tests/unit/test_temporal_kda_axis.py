from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.models.temporal_kda import KDALayoutMetadata, KimiDeltaAttention


def _metadata(values: torch.Tensor) -> KDALayoutMetadata:
    return KDALayoutMetadata(*values.shape)


def test_kda_accepts_only_explicit_b_t_p_d_layout() -> None:
    mixer = KimiDeltaAttention(16, heads=4).eval()
    values = torch.randn(2, 5, 7, 16)
    with pytest.raises(TypeError, match="metadata"):
        mixer(values)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="explicit B,T,P,D metadata"):
        mixer(values, metadata=KDALayoutMetadata(2, 7, 5, 16))


def test_kda_patch_permutation_is_not_a_causal_axis() -> None:
    torch.manual_seed(37)
    mixer = KimiDeltaAttention(16, heads=4).eval()
    values = torch.randn(2, 5, 7, 16)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.argsort(permutation)
    with torch.inference_mode():
        reference = mixer(values, metadata=_metadata(values))
        permuted_values = values[:, :, permutation]
        permuted = mixer(permuted_values, metadata=_metadata(permuted_values))
    assert torch.allclose(reference, permuted[:, :, inverse], atol=1e-5, rtol=1e-5)


def test_kda_future_perturbation_does_not_change_past_outputs() -> None:
    torch.manual_seed(41)
    mixer = KimiDeltaAttention(16, heads=4).eval()
    values = torch.randn(1, 5, 4, 16)
    changed = values.clone()
    changed[:, 4] += 100.0
    with torch.inference_mode():
        reference = mixer(values, metadata=_metadata(values))
        result = mixer(changed, metadata=_metadata(changed))
    assert torch.allclose(reference[:, :4], result[:, :4], atol=1e-5, rtol=1e-5)


def test_kda_padding_holes_are_noops_and_ignore_invalid_values() -> None:
    torch.manual_seed(53)
    mixer = KimiDeltaAttention(16, heads=4).eval()
    values = torch.randn(1, 5, 2, 16)
    valid = torch.ones(1, 5, 2, dtype=torch.bool)
    valid[:, 2, 0] = False
    changed = values.clone()
    changed[:, 2, 0] = 1.0e6
    with torch.inference_mode():
        reference = mixer(values, metadata=_metadata(values), valid_patch_mask=valid)
        result = mixer(changed, metadata=_metadata(changed), valid_patch_mask=valid)
    assert torch.equal(reference[:, 2, 0], torch.zeros_like(reference[:, 2, 0]))
    assert torch.allclose(reference, result, atol=1e-6, rtol=1e-6)
