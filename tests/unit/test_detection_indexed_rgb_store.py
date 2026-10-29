"""Shared ownership and lifecycle tests for indexed RGB dataset views."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from e_jepa_ttc.detection.indexed_rgb_store import IndexedRGBStore


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path: Path) -> tuple[dict[str, tuple[str, str, str]], dict[str, str]]:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    labels = []
    ownership = {}
    with tarfile.open(data / "rgb.tar", "w") as archive:
        for index in range(2):
            token = f"sequence:frame-{index}"
            buffer = io.BytesIO()
            Image.fromarray(np.full((2, 3, 3), index, dtype=np.uint8), "RGB").save(
                buffer, format="PNG"
            )
            raw = buffer.getvalue()
            member = f"rgb/{index}.png"
            info = tarfile.TarInfo(member)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
            ownership[token] = ("data/rgb.tar", member, digest(raw))
            rows.append(
                {
                    "sample_token": token,
                    "sequence_id": "sequence",
                    "split": "train",
                    "rgb_shard_path": "data/rgb.tar",
                    "rgb_member_path": member,
                    "labels_path": "data/labels.parquet",
                    "rgb_exposure_start_timestamp_us": index,
                    "rgb_exposure_end_timestamp_us": index + 1,
                    "K_event": np.eye(3).tolist(),
                    "T_event_ego": np.eye(4).tolist(),
                }
            )
            labels.append(
                {
                    "sample_token": token,
                    "sequence_id": "sequence",
                    "split": "train",
                    "instance_id": index,
                }
            )
    pd.DataFrame(rows).to_parquet(data / "train.parquet", index=False)
    pd.DataFrame(labels).to_parquet(data / "labels.parquet", index=False)
    return ownership, {"data/rgb.tar": digest((data / "rgb.tar").read_bytes())}


def test_views_share_one_store_and_view_close_does_not_close_it(tmp_path: Path) -> None:
    ownership, shards = fixture(tmp_path)
    store = IndexedRGBStore(tmp_path, ownership, shards, max_open_handles=1)
    first = store.dataset_view(["sequence:frame-0"])
    second = store.dataset_view(["sequence:frame-1"])
    assert first[0].sample_token == "sequence:frame-0"
    first.close()
    assert second[0].sample_token == "sequence:frame-1"
    assert store.metrics()["read_count"] == 2
    assert store.open_handle_count == 1
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        second[0]


def test_view_rejects_token_outside_union(tmp_path: Path) -> None:
    ownership, shards = fixture(tmp_path)
    store = IndexedRGBStore(tmp_path, {"sequence:frame-0": ownership["sequence:frame-0"]}, shards)
    with pytest.raises(ValueError, match="absent from shared store"):
        store.dataset_view(["sequence:frame-1"])
    store.close()
