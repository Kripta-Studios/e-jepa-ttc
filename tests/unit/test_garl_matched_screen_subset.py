from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import scripts.build_garl_matched_screen_subset as builder
from e_jepa_ttc.artifacts.hashing import verify_artifact_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    cache_root = tmp_path / "cache"
    (cache_root / "train").mkdir(parents=True)
    (cache_root / "validation").mkdir()
    records = {
        "train": [
            {
                "sequence_id": "train-seq",
                "sample_token": "train-token",
                "track_id": "train-track",
                "public_track_id": "train-track",
                "timestamp_us": torch.tensor(1),
                "ttc_s": torch.tensor(1.0),
            }
        ],
        "validation": [
            {
                "sequence_id": "validation-seq",
                "sample_token": "validation-token",
                "track_id": "validation-track",
                "public_track_id": "validation-track",
                "timestamp_us": torch.tensor(2),
                "ttc_s": torch.tensor(4.0),
            }
        ],
    }
    shards = []
    for split, values in records.items():
        path = cache_root / split / "shard-00000.pt"
        torch.save(values, path)
        shards.append(
            {
                "split": split,
                "path": f"{split}/shard-00000.pt",
                "count": 1,
                "sha256": _sha256(path),
            }
        )
    cache_manifest = cache_root / "manifest.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": "c" * 64,
                "split_counts": {"train": 1, "validation": 1},
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    data_rows = [
        {
            "sequence_id": record["sequence_id"],
            "sample_token": record["sample_token"],
            "track_id": record["track_id"],
            "public_track_id": record["public_track_id"],
            "timestamp_us": int(record["timestamp_us"]),
            "events_path": f"{record['sequence_id']}/events.h5",
        }
        for values in records.values()
        for record in values
    ]
    label_rows = [
        {
            **{key: row[key] for key in builder.JOIN_KEYS},
            "ttc": 1.0 if row["sequence_id"] == "train-seq" else 4.0,
        }
        for row in data_rows
    ]
    data_path = tmp_path / "data.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(data_rows).to_parquet(data_path, index=False)
    pd.DataFrame(label_rows).to_parquet(labels_path, index=False)
    validation_dir = tmp_path / "validation_subset"
    validation_dir.mkdir()
    pd.DataFrame([data_rows[1]]).to_parquet(validation_dir / "data.parquet", index=False)
    validation_manifest = validation_dir / "manifest.json"
    validation_manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": "v" * 64,
                "outputs": {"data": {"path": "data.parquet"}},
            }
        ),
        encoding="utf-8",
    )
    return {
        "cache_manifest": cache_manifest,
        "public_data_parquet": data_path,
        "public_labels_parquet": labels_path,
        "validation_subset_manifest": validation_manifest,
        "output_dir": tmp_path / "output",
    }


def test_build_subset_binds_exact_cache_rows_and_disjoint_sequences(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = builder.build_subset(**paths)

    assert verify_artifact_hash(report)
    assert report["roles"]["train"]["rows"] == 1
    assert report["roles"]["validation"]["rows"] == 1
    assert report["checks"]["train_validation_sequence_disjoint"] is True
    assert report["checks"]["target_equality"] is True
    assert (paths["output_dir"] / "manifest.json").is_file()


def test_build_subset_rejects_cache_target_disagreement(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    labels = pd.read_parquet(paths["public_labels_parquet"])
    labels.loc[labels["sample_token"] == "train-token", "ttc"] = 2.0
    labels.to_parquet(paths["public_labels_parquet"], index=False)

    with pytest.raises(ValueError, match="TTC targets disagree"):
        builder.build_subset(**paths)
