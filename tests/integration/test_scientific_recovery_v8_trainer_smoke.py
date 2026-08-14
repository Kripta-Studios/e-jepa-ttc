"""One-update CPU proof that V8 trains from its own signed temporal cache."""

from __future__ import annotations

import json

import numpy as np

from e_jepa_ttc.data.scientific_recovery_v8_adapter import V8ToObjectEventV4Dataset
from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheConfig,
    write_scientific_recovery_v8_cache_for_testing,
)
from e_jepa_ttc.training.scientific_recovery_v8_trainer import run_v8_cache_smoke


def _record(index: int) -> dict[str, object]:
    return {
        "representation": np.full((3, 6, 8, 8), index + 1, dtype=np.float32),
        "endpoint_us": np.asarray([0, 100_000, 200_000], dtype=np.int64),
        "sample_token": f"token-{index}",
        "sequence_id": f"sequence-{index % 3}",
        "track_id": f"track-{index}",
        "row_identity": [
            f"sequence-{index % 3}",
            f"token-{index}",
            f"track-{index}",
            "p",
            str(index),
        ],
        "target_ttc": np.float32(1.0 + index / 10),
        "sample_weight": np.float32(1.0),
        "outer_fold": np.int64(index % 3),
        "common_roi_xyxy": np.asarray([0, 0, 8, 8], dtype=np.float32),
        "endpoint_boxes_xyxy": np.asarray([[0, 0, 8, 8]] * 3, dtype=np.float32),
        "visible_heights_px": np.asarray([8, 8], dtype=np.float32),
        "representation_source": "fixture",
        "endpoint_diagnostics": [{}, {}, {}],
    }


def test_v8_trainer_smoke_writes_signed_checkpoint_summary_and_predictions(tmp_path) -> None:
    cache = tmp_path / "cache"
    write_scientific_recovery_v8_cache_for_testing(
        records=[_record(index) for index in range(6)],
        output_dir=cache,
        config=ScientificRecoveryV8CacheConfig(
            representation="exp6", roi_size=8, shard_size=6, expected_rows=None
        ),
    )
    result = run_v8_cache_smoke(
        cache_manifest=cache / "manifest.json",
        output_dir=tmp_path / "run",
        outer_fold=0,
        allow_fixture=True,
    )
    assert result["status"] == "completed_train_only_grouped_dev"
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_sha256"]
    assert (tmp_path / "run" / "state" / "last.pt").is_file()
    assert (tmp_path / "run" / "dev_predictions.csv").is_file()


def test_v8_adapter_exposes_real_geometry_to_a5_collate_contract(tmp_path) -> None:
    cache = tmp_path / "cache"
    write_scientific_recovery_v8_cache_for_testing(
        records=[_record(index) for index in range(6)],
        output_dir=cache,
        config=ScientificRecoveryV8CacheConfig(
            representation="exp6", roi_size=8, shard_size=6, expected_rows=None
        ),
    )
    from e_jepa_ttc.data.scientific_recovery_v8_cache import ScientificRecoveryV8CacheDataset

    row = V8ToObjectEventV4Dataset(
        ScientificRecoveryV8CacheDataset(cache / "manifest.json"), outer_fold=0, split="train"
    )[0]
    assert row["event_v4_common_roi"].shape == (3, 6, 8, 8)
    assert row["event_v4_boxes_xyxy"].shape == (3, 4)
    assert row["garl_visible_heights_px"].shape == (2,)
    assert float(row["sample_weight"]) == 1.0
