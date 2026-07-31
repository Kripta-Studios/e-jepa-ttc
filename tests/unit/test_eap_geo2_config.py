from __future__ import annotations

import pytest

from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig


def test_geo2_configuration_is_explicit() -> None:
    config = EAPJEPATrainerConfig(
        geometry_loss_weight=0.25,
        geometry_target_version="v2",
        geometry_sampling_strategy="balanced_tracks",
    )
    assert config.geometry_target_version == "v2"
    assert config.geometry_sampling_strategy == "balanced_tracks"


def test_invalid_geometry_sampling_strategy_fails() -> None:
    with pytest.raises(ValueError):
        EAPJEPATrainerConfig(geometry_sampling_strategy="random")
