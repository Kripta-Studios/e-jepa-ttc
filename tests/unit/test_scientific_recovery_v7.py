from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.eap_representation import event_voxel_with_scalars
from e_jepa_ttc.data.event_v4_common_roi import CommonROIConfig, rasterize_common_roi
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.evaluation.selective_ttc import (
    risk_coverage_curve,
    selective_predictions,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.height_ratio_head import raw_garl_height_ratio_ttc
from e_jepa_ttc.training.causal_scale_eap import (
    CausalScaleEAPTrainingConfig,
    _apply_encoder_freeze,
    _load_soft_geometry_teacher,
)
from scripts.aggregate_v7_fold_results import _paired_bootstrap, _token_sha
from scripts.freeze_scientific_recovery_v7_configs import _sorted_values_sha256
from scripts.train_causal_scale_eap_screen import _validate_partial_freeze_control


def _events() -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        "y": np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        "t": np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        "p": np.asarray([1, -1, 1, -1, 1], dtype=np.int8),
    }


def test_t20_shape_is_finite_and_matches_float32_reference() -> None:
    events = _events()
    config = CommonROIConfig(size=8, bins_per_polarity=10, event_pixel_diff=0)
    raster = rasterize_common_roi(events, (0, 0, 8, 8), start_us=0, end_us=10, config=config)
    batch = EventBatch(
        x=events["x"],
        y=events["y"],
        t_us=events["t"],
        polarity=events["p"],
        width=8,
        height=8,
        sequence_id="fixture",
        t_start_us=0,
        t_end_us=10,
    )
    reference = event_voxel_with_scalars(batch, bins_per_polarity=10)
    assert raster.shape == (22, 8, 8)
    assert torch.isfinite(raster).all()
    torch.testing.assert_close(raster, reference, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(raster.half().float(), reference, rtol=5e-4, atol=5e-4)


def test_default_five_bin_shape_remains_twelve_channels() -> None:
    raster = rasterize_common_roi(
        _events(),
        (0, 0, 8, 8),
        start_us=0,
        end_us=10,
        config=CommonROIConfig(size=8, event_pixel_diff=0),
    )
    assert raster.shape == (12, 8, 8)


def test_adaptive_router_starts_at_ninety_percent_fine_and_uses_six_inputs() -> None:
    config = CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=16,
        geometry_dim=32,
        residual_depth=1,
        transport_enabled=True,
        transport_mode="adaptive_pyramid",
        transport_fine_radius=1,
        transport_coarse_radius=2,
        transport_coarse_downsample=2,
    )
    model = CausalScaleTTC(config)
    assert isinstance(model.transport_router, torch.nn.Sequential)
    assert model.transport_router[1].in_features == 6
    output = model(torch.rand(2, 3, 12, 64, 64), torch.full((2, 2), 0.1))
    torch.testing.assert_close(
        output.diagnostics["transport_fine_weight"],
        torch.full((2, 2), 0.9),
        rtol=0.0,
        atol=1e-6,
    )


def test_old_training_config_loads_without_v7_fields() -> None:
    config = CausalScaleEAPTrainingConfig()
    assert config.freeze_encoder_stages == 0
    assert config.soft_geometry_teacher_checkpoint is None


def test_v7_aggregate_uses_frozen_length_prefixed_token_hash() -> None:
    tokens = ["b", "aa", "a"]
    assert _token_sha(pd.Series(tokens)) == _sorted_values_sha256(tokens)


def test_soft_teacher_is_frozen_in_eval_and_has_no_trainable_parameters(tmp_path) -> None:
    config = CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=16,
        geometry_dim=32,
        residual_depth=1,
    )
    source = CausalScaleTTC(config)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "model_config": source.checkpoint_config(),
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    teacher, metadata = _load_soft_geometry_teacher(
        checkpoint,
        expected_sha256=expected,
        device=torch.device("cpu"),
    )
    assert teacher.training is False
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert metadata["frozen"] is True


def test_partial_freeze_targets_exact_encoder_features_slice() -> None:
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=16,
            geometry_dim=32,
            residual_depth=1,
        )
    )
    frozen = _apply_encoder_freeze(
        model,
        freeze_encoder=False,
        freeze_encoder_stages=3,
    )
    expected = sorted(
        name
        for name, _ in model.named_parameters()
        if name.startswith(("encoder.features.0.", "encoder.features.1.", "encoder.features.2."))
    )
    assert frozen == expected
    assert frozen
    assert all(not dict(model.named_parameters())[name].requires_grad for name in frozen)
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in frozen
    )
    optimizer_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert not optimizer_names.intersection(frozen)


def test_partial_freeze_contract_rejects_layer_or_loss_drift() -> None:
    training = CausalScaleEAPTrainingConfig(
        freeze_encoder_stages=3,
        soft_geometry_teacher_checkpoint="teacher.pt",
        soft_geometry_teacher_checkpoint_sha256="0" * 64,
        soft_dense_cosine_weight=1.0,
        soft_geometry_weight=1.0,
    )
    contract = {
        "partial_freeze_control": {
            "frozen_module_slice": "encoder.features[0:3]",
            "student_initialized_from_scratch": True,
            "student_remaining_layers_trainable": True,
            "same_fold_local_teacher": True,
            "same_soft_loss_weights": True,
            "layer_or_weight_sweep_allowed": False,
        }
    }
    _validate_partial_freeze_control(training, contract)
    with pytest.raises(ValueError, match="loss weights"):
        _validate_partial_freeze_control(
            CausalScaleEAPTrainingConfig(
                freeze_encoder_stages=3,
                soft_geometry_teacher_checkpoint="teacher.pt",
                soft_geometry_teacher_checkpoint_sha256="0" * 64,
                soft_dense_cosine_weight=0.5,
                soft_geometry_weight=1.0,
            ),
            contract,
        )


def test_point_and_abstention_are_separate_when_all_rows_unknown() -> None:
    point = np.asarray([60.0, -60.0])
    selective = selective_predictions(point, np.asarray([False, False]))
    assert np.isfinite(point).all()
    assert np.isnan(selective).all()


def test_risk_coverage_requires_finite_point_predictions() -> None:
    with pytest.raises(ValueError, match="finite"):
        risk_coverage_curve([1.0, 2.0], [1.1, np.nan], [1.0, 0.5], ["a", "b"])


def test_raw_garl_height_ratio_is_singular_for_equal_heights() -> None:
    value = raw_garl_height_ratio_ttc(
        torch.tensor([4.0]),
        torch.tensor([4.0]),
        torch.tensor([0.1]),
    )
    assert torch.isinf(value).all()


def test_bootstrap_rejects_one_sequence_even_with_multiple_tracks() -> None:
    candidate = pd.DataFrame(
        {
            "sample_token": ["a", "b"],
            "sequence_id": ["only", "only"],
            "track_id": ["t0", "t1"],
            "target_ttc_s": [1.0, 2.0],
            "point_prediction_ttc_s": [1.1, 2.1],
        }
    )
    baseline = candidate.assign(point_prediction_ttc_s=[1.2, 2.2])
    with pytest.raises(ValueError, match="at least two sequences"):
        _paired_bootstrap(candidate, baseline, resamples=10, seed=7)
