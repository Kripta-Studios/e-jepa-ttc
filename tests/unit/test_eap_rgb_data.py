"""Exercise real tiny TAR/Parquet inputs and train-only pilot selection."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from e_jepa_ttc.detection.eap_rgb_data import (
    EAPRGBDataset,
    safe_dataset_path,
    safe_relative_path,
    select_pilot_tokens,
)
from scripts.prepare_eap_rgb_pilot import inspect_event_slice


def make_fixture(root: Path, *, duplicate_member: bool = False) -> pd.DataFrame:
    sequence = root / "data/train/a"
    sequence.mkdir(parents=True)
    image = Image.new("RGB", (32, 24), (13, 27, 41))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    raw = encoded.getvalue()
    with tarfile.open(sequence / "frames.tar", "w") as shard:
        for _ in range(2 if duplicate_member else 1):
            member = tarfile.TarInfo("rgb/f.png")
            member.size = len(raw)
            shard.addfile(member, io.BytesIO(raw))
    media = pd.DataFrame(
        [
            {
                "sample_token": "a:f",
                "sequence_id": "a",
                "split": "train",
                "rgb_shard_path": "data/train/a/frames.tar",
                "rgb_member_path": "rgb/f.png",
                "labels_path": "data/train/a/labels.parquet",
                "events_path": "data/train/a/events.h5",
                "rgb_exposure_start_timestamp_us": 1000,
                "rgb_exposure_end_timestamp_us": 1100,
                "K_event": np.eye(3).tolist(),
                "T_event_ego": np.eye(4).tolist(),
            }
        ]
    )
    media.to_parquet(root / "data/train.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_token": "a:f",
                "sequence_id": "a",
                "split": "train",
                "instance_id": "a_1",
                "category": "car",
                "bbox_3d_ego": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
            }
        ]
    ).to_parquet(sequence / "labels.parquet", index=False)
    return media


def test_rgb_is_full_frame_and_raw_labels_are_owned(tmp_path):
    make_fixture(tmp_path)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        sample = dataset[0]
        assert sample.image.shape == (24, 32, 3)
        assert sample.image.dtype == np.uint8
        assert sample.image[0, 0].tolist() == [13, 27, 41]
        assert sample.annotations[0]["bbox_3d_ego"].tolist() == [1, 2, 3, 4, 5, 6, 0]
        sample.annotations[0]["bbox_3d_ego"][0] = 999
        sample.image[:] = 0
        assert dataset[0].annotations[0]["bbox_3d_ego"][0] == 1
        assert dataset[0].image[0, 0].tolist() == [13, 27, 41]
        assert len(sample.member_sha256) == 64
    assert not dataset._shards


def test_arrow_filters_requested_tokens_before_pandas_materialization(tmp_path, monkeypatch):
    make_fixture(tmp_path)
    original = pd.read_parquet
    calls = []

    def inspect(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("filters")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", inspect)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        assert len(dataset[0].annotations) == 1
    assert calls == [
        ("train.parquet", [("sample_token", "in", ["a:f"])]),
        ("labels.parquet", [("sample_token", "in", ["a:f"])]),
    ]


@pytest.mark.parametrize("value", ["../f", "/f", "C:/f", "a\\f", "a/./f", "a//f"])
def test_manifest_path_escape_is_rejected(value):
    with pytest.raises(ValueError):
        safe_relative_path(value)


def test_safe_path_is_under_dataset_root(tmp_path):
    assert safe_dataset_path(tmp_path, "data/train/a.tar") == tmp_path / "data/train/a.tar"


def test_absent_or_duplicate_token_is_rejected(tmp_path):
    make_fixture(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        EAPRGBDataset(tmp_path, ["b:f"])
    with pytest.raises(ValueError, match="unique"):
        EAPRGBDataset(tmp_path, ["a:f", "a:f"])


def test_non_train_media_is_rejected(tmp_path):
    media = make_fixture(tmp_path)
    media["split"] = "test"
    media.to_parquet(tmp_path / "data/train.parquet", index=False)
    with pytest.raises(ValueError, match="only public train"):
        EAPRGBDataset(tmp_path, ["a:f"])


def test_duplicate_tar_member_is_ambiguous(tmp_path):
    make_fixture(tmp_path, duplicate_member=True)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        with pytest.raises(ValueError, match="duplicate member"):
            dataset[0]


def test_reversed_exposure_is_rejected(tmp_path):
    media = make_fixture(tmp_path)
    media["rgb_exposure_end_timestamp_us"] = 999
    media.to_parquet(tmp_path / "data/train.parquet", index=False)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        with pytest.raises(ValueError, match="Reversed exposure"):
            dataset[0]


def test_label_sequence_mismatch_is_rejected(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "data/train/a/labels.parquet"
    labels = pd.read_parquet(path)
    labels["sequence_id"] = "b"
    labels.to_parquet(path, index=False)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        with pytest.raises(ValueError, match="Label sequence"):
            dataset[0]


def test_zero_object_frame_is_preserved(tmp_path):
    make_fixture(tmp_path)
    path = tmp_path / "data/train/a/labels.parquet"
    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    with EAPRGBDataset(tmp_path, ["a:f"]) as dataset:
        assert dataset[0].annotations == []


def selection_fixture():
    assignments = {"train": ["a", "b", "c"], "selection": ["d", "e"], "calibration": ["f"]}
    media = pd.DataFrame(
        [
            {
                "sample_token": f"{sequence}:{index:02d}",
                "sequence_id": sequence,
                "rgb_exposure_start_timestamp_us": index * 1000,
                "split": "train",
            }
            for sequence in "abcdef"
            for index in range(32)
        ]
    )
    return media, assignments


def test_pilot_is_deterministic_and_overfit_is_train_only():
    media, assignments = selection_fixture()
    first = select_pilot_tokens(media, assignments, frames_per_sequence=16)
    second = select_pilot_tokens(
        media.sample(frac=1, random_state=3), assignments, frames_per_sequence=16
    )
    assert first == second
    assert len(first["overfit16"]["sample_tokens"]) == 16
    assert set(first["overfit16"]["sample_tokens"]) <= set(first["train"]["sample_tokens"])
    assert not set(first["train"]["sample_tokens"]) & set(first["selection"]["sample_tokens"])
    assert not set(first["train"]["sequences"]) & set(assignments["calibration"])


def test_pilot_rejects_overlapping_candidate_groups():
    media, assignments = selection_fixture()
    assignments["selection"].append("a")
    with pytest.raises(ValueError, match="overlap"):
        select_pilot_tokens(media, assignments, frames_per_sequence=16)


def test_event_index_bracket_is_trimmed_to_causal_end(tmp_path):
    path = tmp_path / "data/train/a"
    path.mkdir(parents=True)
    timestamps = np.array([1900, 2000, 2500, 6999, 7000, 7100, 7900, 8000], dtype=np.int64)
    with h5py.File(path / "events.h5", "w") as handle:
        handle.create_dataset("events/t", data=timestamps)
        for key in ("x", "y", "p"):
            handle.create_dataset("events/" + key, data=np.ones(len(timestamps), dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=np.array([0, 0, 0, 3, 3, 3, 3, 4, 7, 8]))
    result = inspect_event_slice(
        tmp_path,
        {
            "sequence_id": "a",
            "events_path": "data/train/a/events.h5",
            "rgb_exposure_end_timestamp_us": 7000,
        },
    )
    assert result["removed_before_start"] == 1
    assert result["removed_after_end"] == 2
    assert result["retained_event_count"] == 4
    assert result["future_events_retained"] == 0
    assert result["retained_timestamp_max_us"] == 7000
