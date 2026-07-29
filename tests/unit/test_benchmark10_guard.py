from __future__ import annotations

from pathlib import Path

import pytest

from e_jepa_ttc.data.benchmark10_guard import (
    assert_benchmark_inference_authorized,
    assert_no_sealed_benchmark_paths,
    is_sealed_benchmark_path,
)


def test_training_rejects_sealed_benchmark_without_opening_it() -> None:
    path = Path("datasets/evttc_official_benchmark_sealed")
    assert is_sealed_benchmark_path(path)
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_sealed_benchmark_paths([path])


def test_final_inference_requires_freeze_and_explicit_authorization(tmp_path: Path) -> None:
    sealed = tmp_path / "evttc_official_benchmark_sealed"
    with pytest.raises(PermissionError):
        assert_benchmark_inference_authorized(
            sealed,
            final_freeze_manifest=None,
            explicit_authorization=False,
        )
    freeze = tmp_path / "final_freeze_manifest.json"
    freeze.write_text("{}", encoding="utf-8")
    assert_benchmark_inference_authorized(
        sealed,
        final_freeze_manifest=freeze,
        explicit_authorization=True,
    )
