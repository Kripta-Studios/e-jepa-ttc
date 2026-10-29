from pathlib import Path

import pytest

from e_jepa_ttc.models.object_geo_jepa_ttc import OGEConfig
from e_jepa_ttc.training.object_geo_trainer import (
    _oge_config_from_checkpoint,
    evaluate_object_geo_ttc_checkpoint,
)


def test_checkpoint_config_accepts_known_non_init_audit_field() -> None:
    payload = {
        "in_channels": 21,
        "head_mode": "global",
        "sequence_embedding_forbidden": True,
    }
    restored = _oge_config_from_checkpoint(payload)
    assert isinstance(restored, OGEConfig)
    assert restored.head_mode == "global"
    assert restored.sequence_embedding_forbidden is True


def test_checkpoint_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        _oge_config_from_checkpoint({"invented_architecture_flag": True})


def test_family_holdout_test_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="allow-diagnostic-test"):
        evaluate_object_geo_ttc_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            cache_manifest_path=tmp_path / "missing.json",
            output_dir=tmp_path / "output",
            splits=("test",),
        )


def test_sealed_benchmark_is_rejected_before_checkpoint_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Sealed Benchmark-10"):
        evaluate_object_geo_ttc_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            cache_manifest_path=tmp_path / "evttc_official_benchmark_sealed" / "manifest.json",
            output_dir=tmp_path / "output",
            splits=("validation",),
        )
