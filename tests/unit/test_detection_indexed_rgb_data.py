"""Tests for exact receipt-bound TAR offset RGB reads."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.indexed_rgb_data import IndexedEAPRGBDataset, IndexedTarRGBReader


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_tar(path: Path, members: list[tuple[str, bytes, str]]) -> str:
    with tarfile.open(path, "w") as archive:
        for name, value, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            if kind == "file":
                archive.addfile(info, io.BytesIO(value))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    return digest(path.read_bytes())


def reader(tmp_path: Path, shards: int = 1, cap: int = 2) -> IndexedTarRGBReader:
    ownership = {}
    hashes = {}
    for index in range(shards):
        relative = f"s{index}.tar"
        payload = f"payload-{index}".encode()
        hashes[relative] = write_tar(tmp_path / relative, [("rgb/a.png", payload, "file")])
        ownership[f"t{index}"] = (relative, "rgb/a.png", digest(payload))
    return IndexedTarRGBReader(tmp_path, ownership, hashes, max_open_handles=cap)


def test_direct_reads_reopen_with_independent_handle_limit(tmp_path: Path) -> None:
    indexed = reader(tmp_path, shards=3, cap=2)
    assert indexed.read("t0") == b"payload-0"
    assert indexed.read("t1") == b"payload-1"
    assert indexed.read("t2") == b"payload-2"
    assert indexed.open_handle_count == 2
    assert indexed.read("t0") == b"payload-0"
    assert indexed.open_handle_count == 2
    indexed.close()
    assert indexed.open_handle_count == 0


def test_duplicate_name_rejected_even_when_unrequested(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(
        path,
        [("rgb/a.png", b"a", "file"), ("other.png", b"x", "file"), ("other.png", b"y", "file")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": shard_hash},
        )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b"])
def test_unsafe_unrequested_member_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file"), (name, b"x", "file")])
    with pytest.raises(ValueError, match="Unsafe|Absolute"):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": shard_hash},
        )


def test_nonregular_unrequested_member_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file"), ("folder", b"", "dir")])
    with pytest.raises(ValueError, match="regular"):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": shard_hash},
        )


def test_oversize_unrequested_member_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file"), ("large", b"12345", "file")])
    with pytest.raises(ValueError, match="size limit"):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": shard_hash},
            max_member_bytes=4,
        )


def test_empty_regular_member_allowed_only_when_unrequested(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file"), ("excluded.png", b"", "file")])
    indexed = IndexedTarRGBReader(
        tmp_path,
        {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
        {"s.tar": shard_hash},
    )
    assert indexed.read("t") == b"a"
    with pytest.raises(ValueError, match="empty"):
        IndexedTarRGBReader(
            tmp_path,
            {"bad": ("s.tar", "excluded.png", digest(b""))},
            {"s.tar": shard_hash},
        )


def test_source_mutation_and_truncation_fail_closed(tmp_path: Path) -> None:
    indexed = reader(tmp_path)
    path = tmp_path / "s0.tar"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="identity changed"):
        indexed.read("t0")


def test_malformed_tar_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.tar"
    path.write_bytes(b"not a tar archive")
    with pytest.raises(tarfile.ReadError):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("bad.tar", "rgb/a.png", digest(b"a"))},
            {"bad.tar": digest(path.read_bytes())},
        )


def test_member_digest_and_shard_digest_are_mandatory(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file")])
    with pytest.raises(ValueError, match="source SHA"):
        IndexedTarRGBReader(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": "0" * 64},
        )
    indexed = IndexedTarRGBReader(
        tmp_path,
        {"t": ("s.tar", "rgb/a.png", "0" * 64)},
        {"s.tar": shard_hash},
    )
    with pytest.raises(ValueError, match="differs from admission"):
        indexed.read("t")


def test_token_ownership_is_unique_and_unknown_targets_fail(tmp_path: Path) -> None:
    path = tmp_path / "s.tar"
    shard_hash = write_tar(path, [("rgb/a.png", b"a", "file")])
    with pytest.raises(ValueError, match="multiple token owners"):
        IndexedTarRGBReader(
            tmp_path,
            {
                "one": ("s.tar", "rgb/a.png", digest(b"a")),
                "two": ("s.tar", "rgb/a.png", digest(b"a")),
            },
            {"s.tar": shard_hash},
        )
    indexed = IndexedTarRGBReader(
        tmp_path,
        {"one": ("s.tar", "rgb/a.png", digest(b"a"))},
        {"s.tar": shard_hash},
    )
    with pytest.raises(KeyError, match="not owned"):
        indexed.read("two")


def test_indexed_dataset_has_exact_full_sample_parity(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    image_buffer = io.BytesIO()
    Image.fromarray(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), "RGB").save(
        image_buffer, format="PNG"
    )
    raw = image_buffer.getvalue()
    shard_hash = write_tar(data / "rgb.tar", [("rgb/frame.png", raw, "file")])
    token = "sequence:frame"
    pd.DataFrame(
        [
            {
                "sample_token": token,
                "sequence_id": "sequence",
                "split": "train",
                "rgb_shard_path": "data/rgb.tar",
                "rgb_member_path": "rgb/frame.png",
                "labels_path": "data/labels.parquet",
                "rgb_exposure_start_timestamp_us": 10,
                "rgb_exposure_end_timestamp_us": 20,
                "K_event": np.eye(3).tolist(),
                "T_event_ego": np.eye(4).tolist(),
                "extra_media_field": "preserved",
            }
        ]
    ).to_parquet(data / "train.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_token": token,
                "sequence_id": "sequence",
                "split": "train",
                "instance_id": 7,
                "category": "Car",
                "velocity": np.asarray([1.0, 2.0, 3.0]),
            }
        ]
    ).to_parquet(data / "labels.parquet", index=False)

    with (
        EAPRGBDataset(tmp_path, [token]) as original,
        IndexedEAPRGBDataset(
            tmp_path,
            [token],
            member_sha256_by_token={token: digest(raw)},
            shard_sha256_by_path={"data/rgb.tar": shard_hash},
        ) as indexed,
    ):
        expected = original[0]
        actual = indexed[0]
    assert actual.sample_token == expected.sample_token
    assert actual.sequence_id == expected.sequence_id
    np.testing.assert_array_equal(actual.image, expected.image)
    assert actual.annotations[0].keys() == expected.annotations[0].keys()
    for key, value in actual.annotations[0].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, expected.annotations[0][key])
        else:
            assert value == expected.annotations[0][key]
    np.testing.assert_array_equal(actual.intrinsic, expected.intrinsic)
    np.testing.assert_array_equal(actual.event_from_ego, expected.event_from_ego)
    assert actual.exposure_start_us == expected.exposure_start_us
    assert actual.exposure_end_us == expected.exposure_end_us
    assert actual.media.keys() == expected.media.keys()
    assert actual.member_sha256 == expected.member_sha256 == digest(raw)
