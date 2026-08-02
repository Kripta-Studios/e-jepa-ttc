from pathlib import Path

import pytest

from scripts.train_eap_lhr_jepa_ttc import _config_use_rgb


def test_tubelet_experiment_is_not_silently_trained_as_legacy_model() -> None:
    config = Path("configs/experiment/e_jepa_garl_sota_v1.yaml")
    with pytest.raises(NotImplementedError, match="refusing to silently train"):
        _config_use_rgb(config)


def test_legacy_modality_only_config_remains_supported(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("use_rgb: true\n", encoding="utf-8")
    assert _config_use_rgb(config) is True
