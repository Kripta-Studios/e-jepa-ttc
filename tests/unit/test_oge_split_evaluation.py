from pathlib import Path

import pytest

from e_jepa_ttc.training.object_geo_trainer import evaluate_object_geo_ttc_checkpoint


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
