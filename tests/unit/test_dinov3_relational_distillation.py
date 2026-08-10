"""Tests for A4 DINOv3 dense relational distillation primitives."""

from __future__ import annotations

import torch

from e_jepa_ttc.distillation.dinov3_relational import (
    A4_RELATION_OFFSETS,
    local_cosine_relation_maps,
    local_relational_distillation_loss,
)


def test_relation_maps_identity() -> None:
    # All features are identical across spatial dimensions
    features = torch.ones(2, 3, 32, 32)
    rels = local_cosine_relation_maps(features)
    
    assert rels.values.shape == (2, len(A4_RELATION_OFFSETS), 32, 32)
    assert rels.valid.shape == (2, len(A4_RELATION_OFFSETS), 32, 32)
    assert rels.valid.dtype == torch.bool
    
    valid_mask = rels.valid
    assert valid_mask.any()
    
    # Cosine similarity of identical vectors is 1.0
    valid_values = rels.values[valid_mask]
    assert torch.allclose(valid_values, torch.ones_like(valid_values), atol=1e-5)


def test_relation_maps_orthogonality() -> None:
    # Create alternating orthogonal vectors in spatial grid
    # E.g., features[..., x] and features[..., x+1] are orthogonal
    features = torch.zeros(1, 2, 8, 8)
    for y in range(8):
        for x in range(8):
            if (x + y) % 2 == 0:
                features[0, 0, y, x] = 1.0
            else:
                features[0, 1, y, x] = 1.0

    rels = local_cosine_relation_maps(features)
    
    # Offsets [0, 1] and [1, 0] should give 0 similarity (orthogonal)
    idx_01 = A4_RELATION_OFFSETS.index((0, 1))
    idx_10 = A4_RELATION_OFFSETS.index((1, 0))
    
    valid_01 = rels.valid[0, idx_01]
    assert torch.allclose(rels.values[0, idx_01][valid_01], torch.zeros_like(rels.values[0, idx_01][valid_01]), atol=1e-5)
    
    valid_10 = rels.valid[0, idx_10]
    assert torch.allclose(rels.values[0, idx_10][valid_10], torch.zeros_like(rels.values[0, idx_10][valid_10]), atol=1e-5)
    
    # Offsets [0, 2] and [2, 0] and [1, 1] and [1, -1] should give 1 similarity (identical)
    idx_02 = A4_RELATION_OFFSETS.index((0, 2))
    idx_11 = A4_RELATION_OFFSETS.index((1, 1))
    
    valid_02 = rels.valid[0, idx_02]
    assert torch.allclose(rels.values[0, idx_02][valid_02], torch.ones_like(rels.values[0, idx_02][valid_02]), atol=1e-5)
    
    valid_11 = rels.valid[0, idx_11]
    assert torch.allclose(rels.values[0, idx_11][valid_11], torch.ones_like(rels.values[0, idx_11][valid_11]), atol=1e-5)


def test_relation_maps_border_mask_and_wraparound() -> None:
    features = torch.randn(1, 4, 8, 8)
    rels = local_cosine_relation_maps(features)
    
    # Check bounds explicitly for (dy=1, dx=2) -> not in A4, let's use (0,2) and (1,1)
    for k, (dy, dx) in enumerate(A4_RELATION_OFFSETS):
        for y in range(8):
            for x in range(8):
                y_tgt = y + dy
                x_tgt = x + dx
                is_valid = 0 <= y_tgt < 8 and 0 <= x_tgt < 8
                assert rels.valid[0, k, y, x].item() == is_valid


def test_relation_maps_scale_invariance() -> None:
    features = torch.randn(2, 5, 12, 12)
    rels1 = local_cosine_relation_maps(features)
    rels2 = local_cosine_relation_maps(features * 100.0)
    
    assert torch.allclose(rels1.values[rels1.valid], rels2.values[rels2.valid], atol=1e-5)


def test_relational_distillation_loss() -> None:
    student = torch.randn(2, 2, 3, 16, 16, requires_grad=True)
    teacher_rels = torch.randn(2, 2, 6, 16, 16)
    valid_mask = torch.rand(2, 2, 6, 16, 16) > 0.5
    
    # L1 Loss computation
    loss = local_relational_distillation_loss(student, teacher_rels, valid_mask)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    
    # Gradients
    loss.backward()
    assert student.grad is not None
    assert student.grad.shape == student.shape
    assert not torch.isnan(student.grad).any()


def test_relational_distillation_loss_mixed_precision() -> None:
    student = torch.randn(2, 2, 4, 16, 16, dtype=torch.bfloat16)
    teacher_rels = torch.randn(2, 2, 6, 16, 16, dtype=torch.float16)
    valid_mask = torch.ones(2, 2, 6, 16, 16, dtype=torch.bool)
    
    # The L1 loss should run in float32 internally and return float32
    loss = local_relational_distillation_loss(student, teacher_rels, valid_mask)
    assert loss.dtype == torch.float32
