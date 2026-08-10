"""Tests for A4 DINOv3 model integration and parameter parity."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
)
from e_jepa_ttc.training.causal_scale_eap import CausalScaleEAPTrainingConfig


def test_a4_a1_df_parameter_count_and_identity() -> None:
    # A1-DF uses resize_conv foreground decoder and deep features
    config = CausalScaleTTCConfig(
        foreground_decoder="resize_conv",
    )
    model = CausalScaleTTC(config)

    # 1. Parameter count check
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 355_118

    # 2. Inference identical with/without return_dense_features
    model.eval()
    batch_size, time_steps = 2, 3
    events = torch.randn(batch_size, time_steps, 12, 128, 128)
    delta_t = torch.ones(batch_size, time_steps - 1) * 0.1

    with torch.no_grad():
        out_parent = model(events, delta_t, return_dense_features=False)
        out_dense = model(events, delta_t, return_dense_features=True)

    assert out_parent.endpoint_dense_features is None
    assert out_dense.endpoint_dense_features is not None
    assert out_dense.endpoint_dense_features.shape == (batch_size, time_steps, 64, 32, 32)

    # Scientific outputs must be exactly equal
    assert torch.equal(out_parent.ttc_mean_seconds, out_dense.ttc_mean_seconds)
    assert torch.equal(out_parent.collision_logits, out_dense.collision_logits)
    assert torch.equal(out_parent.foreground_logits, out_dense.foreground_logits)
    assert torch.equal(out_parent.analytic_log_height_ratio, out_dense.analytic_log_height_ratio)


def test_training_config_rejection_rules() -> None:
    # 1. Active supervision without cache SHA fails
    with pytest.raises(ValueError, match="must be provided"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="dinov3_local_relational",
            representation_teacher_cache_artifact_sha256=None,
            representation_distillation_weight=1.0,
        )

    # 2. Active supervision with 0 weight fails
    with pytest.raises(ValueError, match="must be > 0.0"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="dinov3_local_relational",
            representation_teacher_cache_artifact_sha256="dummy_sha",
            representation_distillation_weight=0.0,
        )

    # 3. None supervision with non-zero weight fails
    with pytest.raises(ValueError, match="must be 0.0"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="none",
            representation_distillation_weight=1.0,
        )

    # 4. Valid config succeeds
    valid_cfg = CausalScaleEAPTrainingConfig(
        representation_supervision="dinov3_local_relational",
        representation_teacher_cache_artifact_sha256="dummy_sha",
        representation_distillation_weight=1.0,
    )
    assert valid_cfg.representation_supervision == "dinov3_local_relational"

    # 5. Original A4 rejects a temporal-delta weight: the scientific arm is explicit.
    with pytest.raises(ValueError, match="must be 0.0 unless"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="dinov3_local_relational",
            representation_teacher_cache_artifact_sha256="dummy_sha",
            representation_distillation_weight=1.0,
            representation_temporal_delta_weight=0.5,
        )

    # 6. A4D requires a strictly positive, separately bound temporal weight.
    with pytest.raises(ValueError, match="must be > 0.0"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="dinov3_local_relational_temporal_delta",
            representation_teacher_cache_artifact_sha256="dummy_sha",
            representation_distillation_weight=4.0,
            representation_temporal_delta_weight=0.0,
        )
    with pytest.raises(ValueError, match="calibration artifact identity"):
        CausalScaleEAPTrainingConfig(
            representation_supervision="dinov3_local_relational_temporal_delta",
            representation_teacher_cache_artifact_sha256="dummy_sha",
            representation_distillation_weight=4.0,
            representation_temporal_delta_weight=0.75,
        )
    a4d_cfg = CausalScaleEAPTrainingConfig(
        representation_supervision="dinov3_local_relational_temporal_delta",
        representation_teacher_cache_artifact_sha256="dummy_sha",
        representation_distillation_weight=4.0,
        representation_temporal_delta_weight=0.75,
        representation_temporal_delta_calibration_artifact_sha256="calibration_sha",
    )
    assert a4d_cfg.representation_temporal_delta_weight == 0.75


def test_a4_training_resume_parity(tmp_path) -> None:
    import numpy as np
    from torch.utils.data import Dataset

    from e_jepa_ttc.training.causal_scale_eap import (
        CausalScaleTTCLossConfig,
        train_real_causal_scale,
    )

    class MockDataset(Dataset):
        def __init__(self, has_dino: bool = True):
            self.has_dino = has_dino

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict[str, object]:
            res = {
                "event_v4_common_roi": np.random.default_rng(idx).normal(
                    size=(3, 12, 128, 128)
                ).astype(np.float32),
                "garl_delta_t_s": np.float32(0.1),
                "observable_motion": np.zeros(18, dtype=np.float32),
                "garl_visible_heights_px": np.asarray([20.0, 25.0], dtype=np.float32),
                "ttc_s": np.float32(0.5),
                "event_v4_boxes_xyxy": np.asarray(
                    [[4, 4, 8, 8], [3, 3, 9, 9], [2, 2, 10, 10]], dtype=np.float32
                ),
                "event_v4_common_square_xyxy": np.asarray([0, 0, 128, 128], dtype=np.float32),
                "event_v4_precontext_valid": True,
                "sequence_id": "sequence",
                "sample_token": f"token_{idx}",
                "track_id": "track",
            }
            if self.has_dino:
                # DINO relations (A4)
                res["dinov3_relation_targets"] = np.zeros((2, 6, 32, 32), dtype=np.float16)
                res["dinov3_relation_valid"] = np.ones((2, 6, 32, 32), dtype=np.uint8)
            return res

    model_config = CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=8,
        geometry_dim=16,
        residual_depth=1,
        dropout=0.0,
    )

    training_config = CausalScaleEAPTrainingConfig(
        seed=42,
        epochs=2,
        minimum_epochs=2,
        batch_size=2,
        foreground_warmup_epochs=0,
        representation_supervision="dinov3_local_relational",
        representation_teacher_cache_artifact_sha256="dummy_sha",
        representation_distillation_weight=1.0,
    )

    loss_config = CausalScaleTTCLossConfig(temporal_consistency_weight=0.0)
    device = torch.device("cpu")
    train_ds = MockDataset(has_dino=True)
    val_ds = MockDataset(has_dino=False)

    # 1. Continuous 2 epochs
    res_continuous = train_real_causal_scale(
        model_config,
        training_config,
        loss_config,
        train_ds,
        val_ds,
        device,
    )
    state_continuous = res_continuous.model.state_dict()

    # 2. 1 epoch + save + resume + 1 epoch
    train_real_causal_scale(
        model_config,
        training_config,
        loss_config,
        train_ds,
        val_ds,
        device,
        checkpoint_dir=tmp_path,
        stop_after_epoch=1,
    )

    res_resume = train_real_causal_scale(
        model_config,
        training_config,
        loss_config,
        train_ds,
        val_ds,
        device,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    state_resume = res_resume.model.state_dict()

    # Parity check
    for k in state_continuous:
        torch.testing.assert_close(state_continuous[k], state_resume[k])

    # Negative tests
    bad_cfg1 = replace(
        training_config,
        representation_teacher_cache_artifact_sha256="different_sha",
    )
    with pytest.raises(ValueError, match="resume state differs"):
        train_real_causal_scale(
            model_config,
            bad_cfg1,
            loss_config,
            train_ds,
            val_ds,
            device,
            checkpoint_dir=tmp_path,
            resume=True,
        )

    bad_cfg2 = replace(training_config, representation_distillation_weight=2.0)
    with pytest.raises(ValueError, match="resume state differs"):
        train_real_causal_scale(
            model_config,
            bad_cfg2,
            loss_config,
            train_ds,
            val_ds,
            device,
            checkpoint_dir=tmp_path,
            resume=True,
        )


def test_relational_fg_bg_diagnostic_uses_roi_pixel_coordinates() -> None:
    from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
    from e_jepa_ttc.training.causal_scale_eap import _record_relational_fg_bg_diagnostic

    features = torch.randn(1, 2, 8, 32, 32)
    teacher = local_cosine_relation_maps(features)
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [32.0, 32.0, 96.0, 96.0], [32.0, 32.0, 96.0, 96.0]]]
    )
    components: dict[str, torch.Tensor] = {}
    _record_relational_fg_bg_diagnostic(
        features,
        teacher.values,
        teacher.valid,
        boxes,
        source_height=128,
        source_width=128,
        feat_h=32,
        feat_w=32,
        components=components,
    )
    fraction = float(components["dinov3_relational_fg_fraction"])
    assert 0.15 < fraction < 0.35
