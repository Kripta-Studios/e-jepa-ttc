from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockOuterDevBatch,
    CollisionClockOuterTrainBatch,
    CollisionClockOuterTrainView,
    CollisionClockTrain8192Cache,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import canonical_records_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import PAPER_MID_WEIGHTS

SEQUENCES = [f"sequence-{index}" for index in range(9)]
FOLDS = {sequence: index // 3 for index, sequence in enumerate(SEQUENCES)}
TARGETS = {"negative": -1.0, "crucial": 1.0, "small": 4.0, "large": 8.0}


def _row(index: int, *, wrong_shape: bool = False) -> dict[str, Any]:
    sequence = SEQUENCES[(index // 4) % len(SEQUENCES)]
    bucket, target = list(TARGETS.items())[index % 4]
    del bucket
    shape = (3, 12, 7, 8) if wrong_shape else (3, 12, 8, 8)
    first = 1_000_000 + index * 300_000
    second = first + 100_000
    return {
        "category": "unknown",
        "category_index": np.asarray(0, dtype=np.int64),
        "category_valid": np.asarray(False, dtype=np.bool_),
        "endpoint_delta_error_s": np.asarray(0.0, dtype=np.float32),
        "endpoint_first_timestamp_us": np.asarray(first, dtype=np.int64),
        "endpoint_second_timestamp_us": np.asarray(second, dtype=np.int64),
        "event_v4_boxes_xyxy": np.zeros((3, 4), dtype=np.float32),
        "event_v4_common_roi": np.zeros(shape, dtype=np.float32),
        "event_v4_common_square_xyxy": np.zeros((4,), dtype=np.float32),
        "event_v4_precontext_source": "shifted_event_window_t1_box_proxy",
        "event_v4_precontext_valid": np.asarray(True, dtype=np.bool_),
        "event_v4_t0_box_is_proxy": np.asarray(True, dtype=np.bool_),
        "garl_delta_t_s": np.asarray(0.1, dtype=np.float32),
        "garl_visible_heights_px": np.zeros((2,), dtype=np.float32),
        "geometry_v2_target": np.zeros((20,), dtype=np.float32),
        "geometry_v2_valid": np.zeros((20,), dtype=np.bool_),
        "jepa_context_motion": np.zeros((18,), dtype=np.float32),
        "jepa_pair_valid": np.asarray(True, dtype=np.bool_),
        "observable_motion": np.zeros((18,), dtype=np.float32),
        "precontext_motion_valid": np.asarray(False, dtype=np.bool_),
        "public_track_id": f"track-{index % 6}",
        "sample_token": f"token-{index:03d}",
        "sampling_group": "synthetic",
        "sequence_id": sequence,
        "timestamp_us": np.asarray(second, dtype=np.int64),
        "track_id": f"track-{index % 6}",
        "ttc_label_index": np.asarray(index, dtype=np.int64),
        "ttc_label_source": "synthetic.frame_ttc[t2]",
        "ttc_label_timestamp_us": np.asarray(second, dtype=np.int64),
        "ttc_s": np.asarray(target, dtype=np.float32),
    }


def _fixture_cache(tmp_path: Path, *, wrong_shape: bool = False) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "train8192-mini"
    train = root / "train"
    train.mkdir(parents=True)
    rows = [_row(index, wrong_shape=wrong_shape and index == 0) for index in range(36)]
    shard_rows = [[row] for row in rows]
    for index in range(4):
        shard_rows[index].extend(shard_rows.pop())
    shard_records = []
    flat_rows: list[dict[str, Any]] = []
    for index, values in enumerate(shard_rows):
        relative = f"train/shard-{index:05d}.pt"
        path = root / Path(relative)
        torch.save(values, path)
        flat_rows.extend(values)
        shard_records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "file_sha256": compute_file_hash(str(path)),
                "row_count": len(values),
            }
        )
    manifest_shards = [
        {
            "path": item["path"],
            "size_bytes": item["bytes"],
            "sha256": item["file_sha256"],
            "count": item["row_count"],
            "split": "train",
        }
        for item in shard_records
    ]
    manifest = sign_artifact(
        {
            "artifact_type": "garlttc_official_lhr_object_cache_v4",
            "schema_version": "garlttc_cache_v4",
            "input_schema": {"version": "garlttc_input_v4"},
            "object_lhr_extension": {"event_v4_common_roi_shape": [3, 12, 8, 8]},
            "config": {"delta_t_tolerance_s": 0.025},
            "shards": manifest_shards,
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    frame = pd.DataFrame(
        {
            "sample_token": [str(row["sample_token"]) for row in flat_rows],
            "sequence_id": [str(row["sequence_id"]) for row in flat_rows],
            "track_id": [str(row["track_id"]) for row in flat_rows],
            "target_ttc_s": [float(row["ttc_s"]) for row in flat_rows],
        }
    )
    frame["outer_fold"] = [FOLDS[str(value)] for value in frame["sequence_id"]]
    target_to_bucket = {value: key for key, value in TARGETS.items()}
    frame["sample_weight"] = [
        PAPER_MID_WEIGHTS[target_to_bucket[float(row["ttc_s"])]] / 9.0 for row in flat_rows
    ]
    protocol = {
        "production_row_count": 36,
        "canonical_sequence_to_fold": FOLDS,
        "canonical_bucket_counts_by_sequence": {
            sequence: {bucket: 1 for bucket in TARGETS} for sequence in SEQUENCES
        },
        "canonical_hashes": {
            "token_identity_sha256": canonical_records_hash(
                frame, ("sample_token", "sequence_id", "track_id")
            ),
            "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc_s")),
            "fold_assignment_sha256": canonical_records_hash(
                frame, ("sample_token", "sequence_id", "outer_fold")
            ),
            "sample_weight_sha256": canonical_records_hash(
                frame, ("sample_token", "sample_weight")
            ),
        },
        "cache_binding": {
            "artifact_sha256": manifest["artifact_sha256"],
            "artifact_type": manifest["artifact_type"],
            "bytes": manifest_path.stat().st_size,
            "file_sha256": compute_file_hash(str(manifest_path)),
            "path": "artifacts/cache/train8192-mini/manifest.json",
            "preprocessing_version": "garlttc_input_v4",
            "schema_version": "garlttc_cache_v4",
            "train_shards": shard_records,
        },
    }
    return root, protocol


def test_read_only_adapter_verifies_cache_and_builds_disjoint_typed_views(
    tmp_path: Path,
) -> None:
    root, protocol = _fixture_cache(tmp_path)
    adapter = CollisionClockTrain8192Cache(root, protocol)
    locators = adapter.verify_and_index()
    train, dev = adapter.outer_views(0)
    assert len(locators) == 36
    assert {item.sample_token for item in train.locators}.isdisjoint(
        {item.sample_token for item in dev.locators}
    )
    train_batch = next(adapter.iter_outer_train_batches(train, batch_size=2))
    dev_batch = next(adapter.iter_outer_dev_batches(dev, batch_size=2))
    assert type(train_batch) is CollisionClockOuterTrainBatch
    assert type(dev_batch) is CollisionClockOuterDevBatch
    assert train_batch.inputs.shape == (2, 3, 12, 8, 8)
    assert set(vars(train_batch)) == {
        "inputs",
        "delta_t_s",
        "target_ttc_seconds",
        "sample_tokens",
    }


def test_signed_canonical_supervision_restores_float64_target_identity(tmp_path: Path) -> None:
    root, protocol = _fixture_cache(tmp_path)
    locators = CollisionClockTrain8192Cache(root, protocol).verify_and_index()
    canonical = pd.DataFrame(
        {
            "sample_token": [item.sample_token for item in locators],
            "target_ttc_s": [item.target_ttc_s for item in locators],
            "sample_weight": [item.sample_weight for item in locators],
        }
    )
    canonical.loc[0, "target_ttc_s"] += 1.0e-7
    canonical.loc[0, "sample_weight"] += 1.0e-12
    protocol["canonical_hashes"]["target_sha256"] = canonical_records_hash(
        canonical, ("sample_token", "target_ttc_s")
    )
    protocol["canonical_hashes"]["sample_weight_sha256"] = canonical_records_hash(
        canonical, ("sample_token", "sample_weight")
    )
    adapter = CollisionClockTrain8192Cache(root, protocol, canonical_supervision=canonical)
    restored = adapter.verify_and_index()
    assert restored[0].target_ttc_s == canonical.loc[0, "target_ttc_s"]
    train, _dev = adapter.outer_views(restored[0].outer_fold)
    batch = next(adapter.iter_outer_train_batches(train, batch_size=2))
    assert batch.target_ttc_seconds.dtype == torch.float64


def test_adapter_rejects_missing_or_extra_shard(tmp_path: Path) -> None:
    root, protocol = _fixture_cache(tmp_path)
    extra = root / "train" / "extra.pt"
    torch.save([], extra)
    with pytest.raises(ValueError, match="shard universe"):
        CollisionClockTrain8192Cache(root, protocol).verify_and_index()


def test_adapter_rejects_manifest_declared_shape_mismatch(tmp_path: Path) -> None:
    root, protocol = _fixture_cache(tmp_path, wrong_shape=True)
    with pytest.raises(ValueError, match="shape/dtype"):
        CollisionClockTrain8192Cache(root, protocol).verify_and_index()


def test_adapter_rejects_cache_manifest_sha_mismatch(tmp_path: Path) -> None:
    root, protocol = _fixture_cache(tmp_path)
    protocol["cache_binding"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="physical SHA"):
        CollisionClockTrain8192Cache(root, protocol)


def test_direct_lru_and_preallocated_fold_ram_are_bitwise_equal(tmp_path: Path) -> None:
    root, protocol = _fixture_cache(tmp_path)
    adapter = CollisionClockTrain8192Cache(root, protocol, cache_mode="direct")
    train, _dev = adapter.outer_views(0)
    subset = CollisionClockOuterTrainView(
        0, train.locators[:1], adapter._subset_sha(train.locators[:1])
    )
    direct = adapter._materialize(subset.locators)
    adapter.cache_mode = "shard_lru"
    first = adapter._materialize(subset.locators)
    second = adapter._materialize(subset.locators)
    assert int(adapter.engineering_stats()["shard_cache_hits"]) > 0
    adapter.cache_mode = "fold_ram"
    adapter.stage_view(subset)
    staged = adapter._materialize(subset.locators)
    for expected, lru_value, repeated, staged_value in zip(
        direct, first, second, staged, strict=True
    ):
        assert torch.equal(expected, lru_value)
        assert torch.equal(expected, repeated)
        assert torch.equal(expected, staged_value)
    assert int(adapter.engineering_stats()["staged_rows"]) == 1
