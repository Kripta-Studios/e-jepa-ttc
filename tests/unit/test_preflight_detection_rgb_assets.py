from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from scripts.preflight_detection_rgb_assets import (
    _load_population,
    read_jsonl,
    run_preflight,
    sha256_file,
)


def png_bytes(*, size: tuple[int, int] = (1280, 720)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path, members: dict[str, bytes]) -> dict[str, Path]:
    root = tmp_path / "dataset"
    shard_relative = "data/train/seq/rgb_shards/rgb-00000.tar"
    shard = root / shard_relative
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w:") as archive:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    tokens = [f"seq:{index}" for index in range(len(members))]
    table = pa.table(
        {
            "sample_token": tokens,
            "split": ["train"] * len(tokens),
            "sequence_id": ["seq"] * len(tokens),
            "rgb_shard_path": [shard_relative] * len(tokens),
            "rgb_member_path": list(members),
        }
    )
    media = root / "data/train.parquet"
    pq.write_table(table, media)
    train_manifest = tmp_path / "train.json"
    full_selection = tmp_path / "selection.json"
    write_json(
        train_manifest,
        {
            "schema": "eap_detection_campaign_manifest_v1",
            "roles": {"train": {"sample_tokens": tokens[:1]}},
        },
    )
    write_json(
        full_selection,
        {
            "schema": "eap_detection_full_selection_v1",
            "sample_tokens": tokens[1:],
        },
    )
    inventory = tmp_path / "inventory.json"
    write_json(
        inventory,
        {
            "files": {
                shard_relative: {
                    "size_bytes": shard.stat().st_size,
                    "sha256": sha256_file(shard),
                }
            }
        },
    )
    return {
        "root": root,
        "media": media,
        "train": train_manifest,
        "selection": full_selection,
        "inventory": inventory,
    }


def run(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return run_preflight(
        root=paths["root"],
        train_manifest=paths["train"],
        full_selection=paths["selection"],
        media_path=paths["media"],
        pinned_inventory=paths["inventory"],
        output_dir=output,
    )


def test_complete_good_population_records_member_and_tar_hashes(tmp_path: Path) -> None:
    paths = fixture(tmp_path, {"rgb/a.png": png_bytes(), "rgb/b.png": png_bytes()})
    output = tmp_path / "output"
    summary = run(paths, output)
    assert summary["status"] == "PASS"
    tokens = read_jsonl(output / "token_status.jsonl")
    assert len(tokens) == 2
    assert all(record["status"] == "PASS" for record in tokens)
    assert all(record["image_wh"] == [1280, 720] for record in tokens)
    assert all(len(record["member_sha256"]) == 64 for record in tokens)
    shards = read_jsonl(output / "shard_status.jsonl")
    assert shards[0]["sha256_matches_pinned"] is True


def test_truncated_member_is_retained_while_other_member_continues(tmp_path: Path) -> None:
    paths = fixture(
        tmp_path,
        {"rgb/good.png": png_bytes(), "rgb/truncated.png": png_bytes()[:100]},
    )
    output = tmp_path / "output"
    summary = run(paths, output)
    assert summary["status"] == "FAIL"
    tokens = read_jsonl(output / "token_status.jsonl")
    assert [record["status"] for record in tokens].count("PASS") == 1
    assert [record["status"] for record in tokens].count("FAIL") == 1
    assert summary["bad_tokens"]
    assert (output / "summary.json").is_file()


def test_duplicate_population_token_is_rejected_before_io(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    selection = tmp_path / "selection.json"
    write_json(
        train,
        {
            "schema": "eap_detection_campaign_manifest_v1",
            "roles": {"train": {"sample_tokens": ["same"]}},
        },
    )
    write_json(
        selection,
        {"schema": "eap_detection_full_selection_v1", "sample_tokens": ["same"]},
    )
    with pytest.raises(ValueError, match="Duplicate population tokens"):
        _load_population(train, selection)


def test_pinned_tar_hash_mismatch_fails_shard_but_still_checks_member(tmp_path: Path) -> None:
    paths = fixture(tmp_path, {"rgb/a.png": png_bytes()})
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    relative = next(iter(inventory["files"]))
    inventory["files"][relative]["sha256"] = "0" * 64
    write_json(paths["inventory"], inventory)
    output = tmp_path / "output"
    summary = run(paths, output)
    assert summary["status"] == "FAIL"
    assert read_jsonl(output / "token_status.jsonl")[0]["status"] == "PASS"
    assert read_jsonl(output / "shard_status.jsonl")[0]["status"] == "FAIL"
