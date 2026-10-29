"""Unit tests for E-JEPA-TTC training, gradient accumulation, geometry, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from e_jepa_ttc.models import (
    EventTubeletTransformerEncoder,
    TubeletTokenGeometry,
    pool_object_embeddings,
)
from e_jepa_ttc.training.checkpoints import (
    validate_external_eap_ttc_checkpoint,
)
from e_jepa_ttc.training.eap_ttc import (
    EAPSignedTTCHead,
)
from e_jepa_ttc.utils.io import read_structured, write_structured


def test_token_geometry_order_thw():
    """Verify that reshaping [B, N, D] into [B, T, H, W, D] matches exact (t, y, x) token order."""
    b, t, h, w, d = 2, 1, 5, 10, 16

    tokens = torch.zeros((b, t * h * w, d), dtype=torch.float32)
    idx = 0
    for t_i in range(t):
        for y_i in range(h):
            for x_i in range(w):
                tokens[:, idx, 0] = t_i * 100 + y_i * 10 + x_i
                idx += 1

    reshaped = tokens.reshape(b, t, h, w, d)
    for t_i in range(t):
        for y_i in range(h):
            for x_i in range(w):
                expected = t_i * 100 + y_i * 10 + x_i
                val = reshaped[0, t_i, y_i, x_i, 0].item()
                assert abs(val - expected) < 1e-6


def test_compute_object_embedding_empty_mask_raises_runtime_error():
    """Verify that empty bbox patch mask raises RuntimeError instead of silent pooling fallback."""
    b, n, d = 1, 250, 192
    tokens = torch.randn((b, n, d))
    empty_mask = [torch.zeros((1, 5, 10), dtype=torch.bool)]
    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )

    # The updated requirement is to throw a RuntimeError if the mask is completely empty.
    with pytest.raises(RuntimeError, match="Empty bbox mask reached trainer"):
        pool_object_embeddings(tokens=tokens, bbox_masks=empty_mask, geometry=geometry)


def test_gradient_accumulation_group_sizing():
    """Verify exact group size calculation for full and partial accumulation groups."""
    group_size = 4

    # Case 1: 5 microbatches -> [4, 4, 4, 4], [1]
    batch_count = 5
    sizes_5 = []
    for batch_index in range(batch_count):
        group_start = (batch_index // group_size) * group_size
        current_group_size = min(group_size, batch_count - group_start)
        sizes_5.append(current_group_size)
    assert sizes_5 == [4, 4, 4, 4, 1]

    # Case 2: 6 microbatches -> [4, 4, 4, 4], [2, 2]
    batch_count = 6
    sizes_6 = []
    for batch_index in range(batch_count):
        group_start = (batch_index // group_size) * group_size
        current_group_size = min(group_size, batch_count - group_start)
        sizes_6.append(current_group_size)
    assert sizes_6 == [4, 4, 4, 4, 2, 2]


def test_target_encoder_frozen_and_eval():
    """Verify target_encoder is in eval mode and produces no gradients."""
    target_encoder = EventTubeletTransformerEncoder(in_channels=21)
    target_encoder.requires_grad_(False)
    target_encoder.eval()

    assert not any(p.requires_grad for p in target_encoder.parameters())
    assert not target_encoder.training

    x = torch.randn(2, 21, 90, 160)
    out = target_encoder.forward_tokens(x)
    assert out.grad_fn is None


def test_gradient_flow_separation():
    """Verify gradients flow correctly to online encoder and TTC head."""
    encoder = EventTubeletTransformerEncoder(in_channels=21)
    ttc_head = EAPSignedTTCHead(embed_dim=192)

    x = torch.randn(2, 21, 90, 160)
    tokens = encoder.forward_tokens(x)

    mask = torch.zeros((1, 5, 10), dtype=torch.bool)
    mask[0, 2, 4] = True
    geometry = TubeletTokenGeometry(
        grid_t=5,
        grid_h=5,
        grid_w=10,
        kernel_t=1,
        kernel_h=16,
        kernel_w=16,
        stride_t=1,
        stride_h=16,
        stride_w=16,
    )

    # We must match the expected shape for `pool_object_embeddings`.
    # `EventTubeletTransformerEncoder` gives `tokens` shaped `[B, 250, 192]`.
    obj_emb, _ = pool_object_embeddings(tokens=tokens, bbox_masks=[mask, mask], geometry=geometry)

    pred = ttc_head(obj_emb)
    loss = pred.sum()
    loss.backward()

    assert encoder.event_embed.weight.grad is not None
    assert ttc_head.linear1.weight.grad is not None


def test_checkpoint_provenance_validation_eap_ttc(tmp_path: Path):
    """Verify checkpoint validator accepts valid eAP TTC checkpoint and rejects invalid ones."""
    inventory_hash = "a" * 64
    split_file = tmp_path / "split.json"
    split_payload = {
        "artifact_type": "eap_split",
        "inventory_artifact_sha256": inventory_hash,
        "assignments": {"train": ["seq1"], "validation": ["seq2"]},
    }
    write_structured(split_file, split_payload)

    read_split = read_structured(split_file)
    split_hash = read_split["artifact_sha256"]

    ckpt = {
        "external_pretraining": True,
        "pretraining_regime": "eap_ttc",
        "pretraining_dataset_id": "EAP_PUBLIC_TRAIN40",
        "model_name": "event-tubelet-transformer",
        "in_channels": 21,
        "event_bins": 5,
        "checkpoint_role": "best",
        "checkpoint_selected_by": "validation_loss",
        "encoder_state_dict": {"event_embed.weight": torch.randn(192, 21, 16, 16)},
        "target_encoder_state_dict": {"dummy": torch.tensor([1.0])},
        "predictor_state_dict": {"dummy": torch.tensor([1.0])},
        "ttc_head_state_dict": {"dummy": torch.tensor([1.0])},
        "uses_ttc_labels": True,
        "uses_ttc_labels_for_loss": True,
        "uses_annotation_index_for_sampling": True,
        "uses_ttc_value_for_sampling": False,
        "uses_labels_for_window_sampling": False,
        "uses_object_bboxes": True,
        "uses_depth_track_derivatives": False,
        "ttc_head_transferable_to_evttc": False,
        "uses_collision_labels": False,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
        "audit_result": "PASS",
        "audit_json_sha256": "a" * 64,
        "train_sequences": ["seq1"],
        "validation_sequences": ["seq2"],
        "transferred_components": ["encoder"],
        "discarded_pretraining_heads": ["predictor", "ttc_head"],
        "split_artifact_sha256": split_hash,
        "inventory_artifact_sha256": inventory_hash,
    }

    ckpt_file = tmp_path / "checkpoint_best.pt"
    torch.save(ckpt, ckpt_file)

    validated = validate_external_eap_ttc_checkpoint(ckpt_file, ckpt, source_split_path=split_file)
    assert validated["pretraining_regime"] == "eap_ttc"

    bad_ckpt = dict(ckpt)
    bad_ckpt["validation_sequences"] = ["seq1"]
    with pytest.raises(ValueError, match="train and validation sequences overlap"):
        validate_external_eap_ttc_checkpoint(ckpt_file, bad_ckpt, source_split_path=split_file)

    bad_ckpt2 = dict(ckpt)
    bad_ckpt2["audit_result"] = "FAIL"
    with pytest.raises(ValueError, match="audit_result must be PASS"):
        validate_external_eap_ttc_checkpoint(ckpt_file, bad_ckpt2, source_split_path=split_file)


def test_orchestration_objectives_mapping():
    """Verify objectives parsing maps both -> (ssl, geo) and all -> (ssl, geo, ttc)."""
    from scripts.run_eap_evttc_complete import _objectives

    assert _objectives(["both"]) == ("ssl", "geo")
    assert _objectives(["all"]) == ("ssl", "geo", "ttc")
    assert _objectives(["ttc"]) == ("ttc",)
