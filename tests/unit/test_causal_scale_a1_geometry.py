from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import e_jepa_ttc.training.causal_scale_eap as training
import scripts.train_causal_scale_eap_screen as runner
from e_jepa_ttc.data.object_event_v4 import (
    ObjectEventV4Batch,
    box_geometry_targets,
)
from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    soft_vertical_extent_from_logits,
)

ROOT = Path(__file__).resolve().parents[2]


def _batch() -> ObjectEventV4Batch:
    return ObjectEventV4Batch(
        events=torch.randn(2, 3, 2, 16, 16),
        delta_t_s=torch.full((2,), 0.1),
        observable_motion=torch.zeros(2, 18),
        visible_heights_px=torch.tensor([[8.0, 10.0], [8.0, 7.0]]),
        target_ttc_s=torch.tensor([2.0, -2.0]),
        boxes_xyxy=torch.tensor(
            [
                [[2.0, 3.0, 12.0, 11.0], [2.0, 3.0, 12.0, 11.0], [1.0, 2.0, 13.0, 12.0]],
                [[2.0, 3.0, 12.0, 11.0], [2.0, 3.0, 12.0, 11.0], [3.0, 4.0, 11.0, 11.0]],
            ]
        ),
        common_square_xyxy=torch.tensor([[0.0, 0.0, 16.0, 16.0]]).repeat(2, 1),
        sequence_ids=["seq-a", "seq-b"],
        sample_tokens=["a", "b"],
        track_ids=["track-a", "track-b"],
    )


def _model() -> CausalScaleTTC:
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
            foreground_decoder="equivariant_separable",
            foreground_fullres_dim=8,
        )
    )


class _OneItemDataset(Dataset[int]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        if index != 0:
            raise IndexError(index)
        return index


def test_box_geometry_targets_are_visible_normalized_and_exclude_proxy() -> None:
    boxes = torch.tensor(
        [[[-2.0, 2.0, 6.0, 10.0], [4.0, 3.0, 12.0, 11.0], [4.0, 4.0, 20.0, 12.0]]]
    )
    targets = box_geometry_targets(
        boxes,
        height=16,
        width=16,
        endpoint_valid=torch.tensor([[False, True, True]]),
    )

    assert targets.valid.tolist() == [[False, True, True]]
    assert torch.allclose(
        targets.height_normalized, torch.tensor([[0.0, 0.5, 0.5]])
    )
    assert torch.allclose(
        targets.width_normalized, torch.tensor([[0.0, 0.5, 0.75]])
    )
    assert torch.allclose(
        targets.centroid_x_normalized, torch.tensor([[0.0, 0.5, 0.625]])
    )
    assert torch.allclose(
        targets.centroid_y_normalized, torch.tensor([[0.0, 0.4375, 0.5]])
    )


def test_soft_observation_recovers_width_height_and_centroid() -> None:
    logits = torch.full((1, 1, 32, 40), -20.0)
    logits[..., 8:24, 5:25] = 20.0

    observation = soft_vertical_extent_from_logits(logits)

    assert observation.height_normalized.item() == pytest.approx(16 / 32, abs=1e-5)
    assert observation.width_normalized.item() == pytest.approx(20 / 40, abs=1e-5)
    assert observation.centroid_x_normalized.item() == pytest.approx(15 / 40, abs=1e-5)
    assert observation.centroid_y_normalized.item() == pytest.approx(16 / 32, abs=1e-5)


def test_a1_targets_do_not_rasterize_or_enter_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("A1 attempted to rasterize a weak-box mask")

    monkeypatch.setattr(training, "weak_box_masks", forbidden)
    targets = training._targets(
        _batch(),
        mask_t0_as_proxy=True,
        foreground_supervision="bbox_geometry",
    )

    assert targets.target_masks is None
    assert targets.mask_valid is None
    assert targets.geometry is not None
    assert targets.geometry.valid.tolist() == [[False, True, True], [False, True, True]]
    assert tuple(inspect.signature(CausalScaleTTC.forward).parameters) == (
        "self",
        "inputs",
        "delta_t_s",
        "return_dense_features",
    )


def test_a1_geometry_loss_is_dense_mask_free_and_reaches_foreground() -> None:
    model = _model().train()
    batch = _batch()
    targets = training._targets(
        batch,
        mask_t0_as_proxy=True,
        foreground_supervision="bbox_geometry",
    )
    delta = batch.delta_t_s[:, None].expand(-1, 2)
    output = model(batch.events, delta)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_geometry=targets.geometry,
        config=CausalScaleTTCLossConfig(
            foreground_bce_weight=0.0,
            foreground_dice_weight=0.0,
            foreground_extent_weight=1.25,
            foreground_width_weight=1.25,
            foreground_center_weight=2.5,
            foreground_pair_ratio_weight=0.0,
        ),
    )

    assert result.counts["foreground"] == 0
    assert result.counts["foreground_geometry"] == 4
    assert result.components["foreground_bce"].item() == 0.0
    assert result.components["foreground_dice"].item() == 0.0
    assert torch.isfinite(result.total)
    result.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.encoder.foreground.parameters()
        if parameter.requires_grad
    ]
    assert any(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )


def test_a1_pair_ratio_uses_numeric_geometry_without_dense_masks() -> None:
    model = _model().train()
    batch = _batch()
    targets = training._targets(
        batch,
        mask_t0_as_proxy=True,
        foreground_supervision="bbox_geometry",
    )
    delta = batch.delta_t_s[:, None].expand(-1, 2)
    output = model(batch.events, delta)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_geometry=targets.geometry,
        config=CausalScaleTTCLossConfig(
            log_ratio_nll_weight=0.0,
            log_ratio_huber_weight=0.0,
            log_ratio_tail_weight=0.0,
            foreground_bce_weight=0.0,
            foreground_dice_weight=0.0,
            foreground_extent_weight=0.0,
            foreground_width_weight=0.0,
            foreground_center_weight=0.0,
            foreground_pair_ratio_weight=1.0,
            risk_weight=0.0,
            auxiliary_inverse_ttc_weight=0.0,
            residual_regularization_weight=0.0,
            temporal_consistency_weight=0.0,
        ),
    )

    assert result.counts["foreground_pair_ratio"] == 2
    assert result.components["foreground_pair_ratio"].item() > 0.0
    assert result.total.item() == result.components["foreground_pair_ratio"].item()
    result.total.backward()
    gradients = [
        parameter.grad
        for parameter in model.encoder.foreground.parameters()
        if parameter.requires_grad
    ]
    assert any(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )


def test_pair_ratio_is_disabled_during_geometry_warmup() -> None:
    configured = CausalScaleTTCLossConfig(
        foreground_pair_ratio_weight=5.0,
        log_ratio_nll_weight=1.0,
    )

    warmup = training._foreground_only_loss_config(configured)

    assert warmup.foreground_pair_ratio_weight == 0.0
    assert configured.foreground_pair_ratio_weight == 5.0


def test_geometry_diagnostics_are_global_and_macro_sequence_without_nan_filling() -> None:
    target = np.asarray([0.0, 1.0, 0.0, 2.0])
    prediction = np.asarray([0.0, 2.0, 0.0, 4.0])
    sequences = np.asarray(["a", "a", "b", "b"])

    metrics = training._relationship_by_sequence(target, prediction, sequences)

    assert metrics["global"]["pearson"] == pytest.approx(1.0)
    assert metrics["global"]["slope"] == pytest.approx(2.0)
    assert metrics["macro_by_sequence"]["slope"] == pytest.approx(2.0)
    assert metrics["macro_by_sequence"]["sequence_count"] == 2


def test_a1_evaluation_reports_absolute_and_differential_geometry() -> None:
    loader = DataLoader(
        _OneItemDataset(),
        batch_size=1,
        collate_fn=lambda _: _batch(),
    )
    config = training.CausalScaleEAPTrainingConfig(
        epochs=2,
        minimum_epochs=2,
        foreground_warmup_epochs=1,
        foreground_supervision="bbox_geometry",
        precision="fp32",
    )
    metrics = training.evaluate_real_causal_scale(
        _model().eval(),
        cast(DataLoader[ObjectEventV4Batch], loader),
        torch.device("cpu"),
        config,
        CausalScaleTTCLossConfig(
            foreground_bce_weight=0.0,
            foreground_dice_weight=0.0,
            foreground_extent_weight=1.25,
            foreground_width_weight=1.25,
            foreground_center_weight=2.5,
        ),
    )

    diagnostics = metrics["geometry_diagnostics"]
    assert diagnostics is not None
    assert metrics["weak_bbox_iou_count"] == 0
    assert diagnostics["absolute_log_height"]["global"]["count"] == 4
    assert diagnostics["delta_log_height_vs_bbox"]["global"]["count"] == 2
    assert diagnostics["delta_log_width_vs_physical"]["global"]["count"] == 2
    assert diagnostics["r_iso_is_diagnostic_only"] is True


def test_a1_preregistered_config_keeps_a0_model_and_only_geometry_supervision() -> None:
    config_path = (
        ROOT
        / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_geometry_v1.yaml"
    )
    raw = runner._read_yaml(config_path)
    model_path = ROOT / str(raw["model_config"])
    model_config = runner._model_config(model_path)
    training_config = training.CausalScaleEAPTrainingConfig(**raw["training"])
    loss_config = CausalScaleTTCLossConfig(**raw["loss"])

    assert training_config.foreground_supervision == "bbox_geometry"
    assert loss_config.foreground_bce_weight == 0.0
    assert loss_config.foreground_dice_weight == 0.0
    assert loss_config.foreground_pair_ratio_weight == 0.0
    assert loss_config.foreground_extent_weight == 1.25
    assert loss_config.foreground_width_weight == 1.25
    assert loss_config.foreground_center_weight == 2.5
    parameter_count = sum(
        parameter.numel() for parameter in CausalScaleTTC(model_config).parameters()
    )
    assert parameter_count == 344591
    assert runner._sha256(model_path) == raw["decision_contract"]["model_config_sha256"]
