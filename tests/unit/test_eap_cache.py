from __future__ import annotations

import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from e_jepa_ttc.data.eap_cache import (
    EAPObjectCacheConfig,
    EAPObjectCacheDataset,
    ShardLocalSampler,
    materialize_eap_object_cache,
)


def _write_synthetic_eap(root: Path) -> None:
    sequence = "synthetic"
    sequence_root = root / "data" / "train" / sequence
    sequence_root.mkdir(parents=True)
    # eAP frame intervals are not exact round millisecond multiples. The 21 us
    # offset exercises causal truncation of a nominal 100 ms future window.
    frame_times = 1_000_000 + np.arange(11, dtype=np.int64) * 99_979
    tokens = [f"{sequence}:{timestamp}" for timestamp in frame_times]
    intrinsic = np.asarray(
        [[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]]
    )
    event_from_ego = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    media = pd.DataFrame(
        {
            "sample_token": tokens,
            "split": ["train"] * len(tokens),
            "sequence_id": [sequence] * len(tokens),
            "rgb_shard_path": [f"data/train/{sequence}/rgb.tar"] * len(tokens),
            "rgb_member_path": [f"rgb/{token}.png" for token in tokens],
            "events_path": [f"data/train/{sequence}/events.h5"] * len(tokens),
            "labels_path": [f"data/train/{sequence}/labels.parquet"] * len(tokens),
            "rgb_exposure_start_timestamp_us": frame_times,
            "rgb_exposure_end_timestamp_us": frame_times,
            "K_event": [intrinsic.tolist()] * len(tokens),
            "T_event_ego": [event_from_ego.tolist()] * len(tokens),
        }
    )
    (root / "data").mkdir(exist_ok=True)
    media.to_parquet(root / "data" / "train.parquet")
    labels = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": [sequence] * len(tokens),
            "track_id": ["track"] * len(tokens),
            "category": ["car"] * len(tokens),
            "bbox_3d_ego": [
                [20.0 - 10.0 * (timestamp - 1_000_000) * 1e-6, 0, 0, 2, 2, 2, 0]
                for timestamp in frame_times
            ],
        }
    )
    labels.to_parquet(sequence_root / "labels.parquet")
    with tarfile.open(sequence_root / "rgb.tar", "w") as archive:
        for index, token in enumerate(tokens):
            payload = io.BytesIO()
            Image.new("RGB", (1280, 720), color=(index, 20, 30)).save(payload, format="PNG")
            data = payload.getvalue()
            member = tarfile.TarInfo(f"rgb/{token}.png")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    timestamps = np.arange(0, 2_101_000, 1000, dtype=np.int64)
    with h5py.File(sequence_root / "events.h5", "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.full(timestamps.shape, 640, dtype=np.uint16))
        events.create_dataset("y", data=np.full(timestamps.shape, 360, dtype=np.uint16))
        events.create_dataset("t", data=timestamps)
        events.create_dataset("p", data=(np.arange(timestamps.size) % 2).astype(np.int8))
        handle.create_dataset(
            "ms_to_idx",
            data=np.searchsorted(timestamps, np.arange(2102) * 1000).astype(np.uint64),
        )


def test_materialized_eap_object_cache_is_disjoint_and_loadable(tmp_path: Path) -> None:
    root = tmp_path / "eap"
    output = tmp_path / "cache"
    _write_synthetic_eap(root)
    config = EAPObjectCacheConfig(
        roi_width=16,
        roi_height=16,
        event_bins=2,
        shard_size=2,
        include_rgb=True,
        rgb_width=12,
        rgb_height=10,
    )

    manifest = materialize_eap_object_cache(
        eap_root=root,
        output_dir=output,
        sequence_splits={"synthetic": "train"},
        config=config,
        max_windows_per_sequence=3,
    )

    assert manifest["total_samples"] == 3
    dataset = EAPObjectCacheDataset(output / "manifest.json", splits=("train",))
    sample = dataset[0]
    assert len(dataset) == 3
    assert sample["context_events"].shape == (3, 4, 16, 16)
    assert sample["future_events"].shape == (3, 4, 16, 16)
    assert sample["context_rgb"].shape == (3, 3, 10, 12)
    assert sample["context_rgb"].dtype == torch.uint8
    assert sample["context_boxes"].shape == (3, 1, 4)
    valid_future = sample["future_object_mask"].squeeze(-1).numpy().astype(bool)
    assert np.all(
        sample["future_window_start_us"].numpy()[valid_future]
        >= sample["context_window_end_us"].numpy()[-1]
    )
    assert sample["future_window_start_us"].numpy()[0] == sample[
        "context_window_end_us"
    ].numpy()[-1]
    assert not sample["future_ego_action_mask"].any()
    assert sample["ttc_source"].startswith("reconstructed_public")
    order = list(ShardLocalSampler(dataset, seed=7))
    shard_order = [dataset.shard_index(index) for index in order]
    # Every compressed shard is visited in one contiguous run even though both
    # shard order and within-shard sample order are shuffled.
    runs = [shard_order[0]]
    runs.extend(
        shard
        for previous, shard in zip(shard_order, shard_order[1:], strict=False)
        if shard != previous
    )
    assert len(runs) == len(set(runs)) == len(dataset.shard_paths)
    dataset.close()


def test_density_adaptive_roi_window_is_causal_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "eap"
    output = tmp_path / "adaptive-cache"
    _write_synthetic_eap(root)
    manifest = materialize_eap_object_cache(
        eap_root=root,
        output_dir=output,
        sequence_splits={"synthetic": "test"},
        config=EAPObjectCacheConfig(
            roi_width=16,
            roi_height=16,
            event_bins=2,
            adaptive_event_count=20,
            minimum_adaptive_window_ms=10,
        ),
        max_windows_per_sequence=1,
    )

    assert manifest["event_window_policy"] == "roi_density_adaptive_trailing_count"
    dataset = EAPObjectCacheDataset(output / "manifest.json", splits=("test",))
    sample = dataset[0]
    duration_us = (
        sample["context_window_end_us"] - sample["context_window_start_us"]
    ).numpy()
    assert np.all(duration_us >= 10_000)
    assert np.all(duration_us < 100_000)
    valid_future = sample["future_object_mask"].squeeze(-1).numpy().astype(bool)
    assert np.all(
        sample["future_window_start_us"].numpy()[valid_future]
        >= sample["context_window_end_us"].numpy()[-1]
    )
    dataset.close()
