from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_eap import (
    CausalScaleEAPTrainingConfig,
    train_real_causal_scale,
)


class _TinyRealDataset(Dataset[dict[str, Any]]):
    def __init__(self, *, seed: int, sequence_id: str) -> None:
        generator = torch.Generator().manual_seed(seed)
        targets = (1.5, 4.5, 8.0, -2.0) * 2
        self.records: list[dict[str, Any]] = []
        for index, target in enumerate(targets):
            events = torch.rand((3, 12, 16, 16), generator=generator)
            events[1] *= 1.0 + 0.03 * (index + 1)
            events[2] *= 1.0 + 0.06 * (index + 1)
            boxes = torch.tensor(
                [[3.0, 3.0, 12.0, 12.0], [3.0, 3.0, 12.0, 12.0], [2.0, 2.0, 13.0, 13.0]]
            )
            self.records.append(
                {
                    "event_v4_common_roi": events,
                    "garl_delta_t_s": 0.1,
                    "observable_motion": torch.zeros(18),
                    "garl_visible_heights_px": torch.tensor([9.0, 11.0]),
                    "ttc_s": target,
                    "event_v4_boxes_xyxy": boxes,
                    "event_v4_common_square_xyxy": torch.tensor([0.0, 0.0, 16.0, 16.0]),
                    "sequence_id": sequence_id,
                    "sample_token": f"{sequence_id}-{index}",
                    "track_id": f"track-{index}",
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _model_config() -> CausalScaleTTCConfig:
    return CausalScaleTTCConfig(
        in_channels=12,
        hidden_dim=16,
        geometry_dim=24,
        residual_depth=1,
        dropout=0.0,
        foreground_decoder="equivariant_separable",
        foreground_temporal_smoothing=0.15,
        min_abs_log_ratio=1.0e-8,
        min_sensor_support=0.0,
    )


def _training_config() -> CausalScaleEAPTrainingConfig:
    return CausalScaleEAPTrainingConfig(
        seed=19,
        epochs=4,
        minimum_epochs=4,
        early_stopping_patience=4,
        foreground_warmup_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-4,
        minimum_learning_rate=1.0e-5,
        num_workers=0,
        precision="fp32",
        maximum_runtime_hours=1.0,
    )


def _assert_state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_real_causal_scale_resume_matches_uninterrupted_epochs(tmp_path: Path) -> None:
    train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
    validation = _TinyRealDataset(seed=202, sequence_id="validation-sequence")
    device = torch.device("cpu")
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)

    uninterrupted = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "uninterrupted",
    )
    interrupted = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "resumed",
        stop_after_epoch=2,
    )
    assert len(interrupted.history) == 2
    resumed = train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        device,
        checkpoint_dir=tmp_path / "resumed",
        resume=True,
    )

    assert len(uninterrupted.history) == len(resumed.history) == 4
    assert uninterrupted.best_epoch == resumed.best_epoch
    assert uninterrupted.best_selection == resumed.best_selection
    _assert_state_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    full_saved = torch.load(
        tmp_path / "uninterrupted" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved = torch.load(
        tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False
    )
    assert saved["epoch"] == 4
    assert saved["best_epoch"] == resumed.best_epoch
    assert len(saved["history"]) == 4
    for key in (
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_model_state_dict",
        "best_selection",
        "best_epoch",
        "stale",
        "loader_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "python_random_state",
        "numpy_random_state",
    ):
        _assert_nested_equal(full_saved[key], saved[key])
    for full_epoch, resumed_epoch in zip(
        full_saved["history"], saved["history"], strict=True
    ):
        _assert_nested_equal(
            {key: value for key, value in full_epoch.items() if key != "elapsed_seconds"},
            {
                key: value
                for key, value in resumed_epoch.items()
                if key != "elapsed_seconds"
            },
        )


def test_real_causal_scale_resume_rejects_changed_contract(tmp_path: Path) -> None:
    train = _TinyRealDataset(seed=101, sequence_id="train-sequence")
    validation = _TinyRealDataset(seed=202, sequence_id="validation-sequence")
    state = tmp_path / "state"
    loss = CausalScaleTTCLossConfig(log_ratio_tail_weight=2.0)
    train_real_causal_scale(
        _model_config(),
        _training_config(),
        loss,
        train,
        validation,
        torch.device("cpu"),
        checkpoint_dir=state,
        stop_after_epoch=2,
    )
    changed = CausalScaleEAPTrainingConfig(
        **{**_training_config().__dict__, "learning_rate": 2.0e-4}
    )

    try:
        train_real_causal_scale(
            _model_config(),
            changed,
            loss,
            train,
            validation,
            torch.device("cpu"),
            checkpoint_dir=state,
            resume=True,
        )
    except ValueError as error:
        assert "config/data contract" in str(error)
    else:
        raise AssertionError("changed resume contract was accepted")
