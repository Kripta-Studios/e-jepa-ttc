from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.garl_release_cache import (
    RELEASE_CACHE_ARTIFACT,
    GarlReleaseCacheConfig,
    GarlReleaseCacheDataset,
    GarlReleaseShardBatchSampler,
)


def _write_manifest(tmp_path):
    first = tmp_path / "train" / "shard-00000.npz"
    second = tmp_path / "train" / "shard-00001.npz"
    first.parent.mkdir(parents=True)
    common = {
        "ttc_s": np.asarray([1.5, 2.5], dtype=np.float32),
        "visible_height": np.ones((2, 2), dtype=np.float32),
        "delta_t_s": np.asarray([0.1, 0.1], dtype=np.float32),
        "sequence_id": np.asarray(["seq-a", "seq-a"]),
        "sample_token": np.asarray(["a-0", "a-1"]),
        "track_id": np.asarray(["track", "track"]),
        "public_track_id": np.asarray(["public", "public"]),
        "timestamp_us": np.asarray([100, 200], dtype=np.int64),
    }
    np.savez_compressed(
        first,
        event_q=np.full((2, 20, 4, 4), 32768, np.uint16),
        rgb_f16=np.full((2, 2, 3, 4, 4), 1.5, np.float16),
        **common,
    )
    second_common = {key: value[:1] for key, value in common.items()}
    second_common["sequence_id"] = np.asarray(["seq-b"])
    second_common["sample_token"] = np.asarray(["b-0"])
    np.savez_compressed(
        second,
        event_q=np.full((1, 20, 4, 4), 65535, np.uint16),
        rgb_f16=np.full((1, 2, 3, 4, 4), -1.5, np.float16),
        **second_common,
    )
    manifest = {
        "artifact_type": RELEASE_CACHE_ARTIFACT,
        "shards": [
            {"split": "train", "path": "train/shard-00000.npz", "count": 2},
            {"split": "train", "path": "train/shard-00001.npz", "count": 1},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_release_cache_config_rejects_non_official_bins() -> None:
    with pytest.raises(ValueError, match="10 bins"):
        GarlReleaseCacheConfig(roi_bins=5)


def test_release_cache_dataset_decodes_quantized_arrays(tmp_path) -> None:
    manifest = _write_manifest(tmp_path)
    dataset = GarlReleaseCacheDataset(manifest, split="train")

    first = dataset[0]
    last = dataset[2]

    assert first["event_roi"].shape == (20, 4, 4)
    assert torch.allclose(first["event_roi"], torch.full((20, 4, 4), 32768 / 65535))
    assert torch.equal(last["event_roi"], torch.ones((20, 4, 4)))
    assert first["rgb_pair"].shape == (2, 3, 4, 4)
    assert torch.equal(last["rgb_pair"], torch.full((2, 3, 4, 4), -1.5))
    assert first["sample_token"] == "a-0"
    assert last["sample_token"] == "b-0"


def test_release_cache_dataset_can_decode_only_selected_model_fields(tmp_path) -> None:
    manifest = _write_manifest(tmp_path)
    dataset = GarlReleaseCacheDataset(
        manifest,
        split="train",
        fields=("ttc_s", "visible_height", "delta_t_s", "sequence_id", "sample_token", "rgb_f16"),
    )

    row = dataset[0]

    assert "rgb_pair" in row
    assert "event_roi" not in row


def test_release_cache_dataset_accepts_legacy_worker_without_fields_attribute(tmp_path) -> None:
    dataset = GarlReleaseCacheDataset(_write_manifest(tmp_path), split="train")
    del dataset.fields

    row = dataset[0]

    assert "event_roi" in row
    assert "rgb_pair" in row


def test_release_shard_sampler_keeps_each_batch_in_one_shard(tmp_path) -> None:
    dataset = GarlReleaseCacheDataset(_write_manifest(tmp_path), split="train")
    sampler = GarlReleaseShardBatchSampler(dataset, batch_size=2, seed=7)

    batches = list(iter(sampler))

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]
    assert all(len({dataset.shard_index(index) for index in batch}) == 1 for batch in batches)
    assert len({dataset.shard_index(batch[0]) for batch in batches}) == len(batches)
