from __future__ import annotations

import json
import time

import pytest

from e_jepa_ttc.training.eap_jepa import (
    EAPJEPATrainerConfig,
    _write_training_progress,
)


def test_progress_sidecar_is_atomic_and_reports_batch_fraction(tmp_path) -> None:
    path = tmp_path / "run" / "progress.json"

    _write_training_progress(
        path,
        status="running",
        stage="train",
        epoch=2,
        batch_index=4,
        batch_count=8,
        samples_processed=12,
        sample_count=24,
        started=time.perf_counter() - 1.0,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "eap_jepa_progress_v1"
    assert payload["status"] == "running"
    assert payload["stage"] == "train"
    assert payload["epoch"] == 2
    assert payload["batch_index"] == 4
    assert payload["fraction"] == pytest.approx(0.5)
    assert not path.with_name(".progress.json.tmp").exists()


def test_progress_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="integer controls"):
        EAPJEPATrainerConfig(progress_interval_batches=0)


def test_temporal_voxel_cache_capacity_is_positive_and_configurable() -> None:
    assert EAPJEPATrainerConfig().temporal_voxel_cache_size == 16
    assert EAPJEPATrainerConfig(temporal_voxel_cache_size=64).temporal_voxel_cache_size == 64
    with pytest.raises(ValueError, match="integer controls"):
        EAPJEPATrainerConfig(temporal_voxel_cache_size=0)
