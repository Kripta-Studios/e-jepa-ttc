from __future__ import annotations

from pathlib import Path

import pytest

from e_jepa_ttc.data.storage_guard import (
    DatasetId,
    StorageBudget,
    StorageBudgetError,
    assert_bounded_cache_request,
    assert_no_forbidden_cache_features,
    assert_storage_budget,
    estimate_dense_voxel_cache_bytes,
    validate_closed_dataset_roots,
)


def test_full_dataset_voxel_cache_is_forbidden() -> None:
    with pytest.raises(StorageBudgetError, match="Full-dataset voxel caches"):
        assert_bounded_cache_request(None)
    assert assert_bounded_cache_request(64) == 64


def test_preflight_rejects_projected_cache_over_budget(tmp_path: Path) -> None:
    with pytest.raises(StorageBudgetError, match="exceeds the hard budget"):
        assert_storage_budget(
            tmp_path,
            budget=StorageBudget(maximum_cache_gib=0.001, minimum_free_gib=0),
            planned_write_bytes=2 * 1024**2,
        )


def test_dense_cache_estimate_is_uncompressed_and_conservative() -> None:
    estimate = estimate_dense_voxel_cache_bytes(
        samples=32 * 64,
        frames_per_sample=6,
        channels=10,
        height=90,
        width=160,
    )
    assert 6 * 1024**3 < estimate < 8 * 1024**3


def test_closed_registry_rejects_unknown_duplicate_and_benchmark_training(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        validate_closed_dataset_roots({"OTHER": tmp_path / "other"}, for_training=False)
    with pytest.raises(ValueError, match="independent"):
        validate_closed_dataset_roots(
            {
                DatasetId.EVTTC32_LABELLED: tmp_path,
                DatasetId.EAP_HF_TRAIN40: tmp_path,
            },
            for_training=False,
        )
    with pytest.raises(ValueError, match="cannot be registered"):
        validate_closed_dataset_roots(
            {DatasetId.BENCHMARK10_SEALED: tmp_path / "sealed"},
            for_training=True,
        )


def test_forbidden_teacher_and_voxel_products_are_rejected() -> None:
    with pytest.raises(StorageBudgetError, match="Forbidden cache"):
        assert_no_forbidden_cache_features(["sam_full_resolution_logits"])
