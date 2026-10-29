from __future__ import annotations

import inspect

import pytest

from e_jepa_ttc.training.object_geo_trainer import OGETrainerConfig
from e_jepa_ttc.training.supervised import train_tiny_cnn


def test_oge_trainer_has_fast_early_stopping_defaults() -> None:
    config = OGETrainerConfig()
    assert config.early_stopping_patience > 0
    assert config.early_stopping_min_epochs < config.epochs
    assert config.precision == "bf16"
    assert config.num_workers > 0


def test_supervised_base_exposes_early_stop_resume_and_workers() -> None:
    parameters = inspect.signature(train_tiny_cnn).parameters
    for name in (
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_delta_relative",
        "resume",
        "num_workers",
    ):
        assert name in parameters


def test_invalid_training_controls_fail_before_gpu_work() -> None:
    with pytest.raises(ValueError):
        OGETrainerConfig(gradient_accumulation=0)
    with pytest.raises(ValueError):
        OGETrainerConfig(precision="tf32")
