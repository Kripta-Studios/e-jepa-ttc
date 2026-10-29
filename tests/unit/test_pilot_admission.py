from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e_jepa_ttc.detection.pilot_admission import validate_pilot_admission

REVISION = "a" * 40


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row(token: str, *, split: str = "train", sequence: str | None = None) -> dict:
    sequence = sequence or token.split(":", 1)[0]
    stem = token.split(":", 1)[1]
    root = f"data/train/{sequence}"
    return {
        "sample_token": token,
        "split": split,
        "sequence_id": sequence,
        "rgb_shard_path": f"{root}/rgb_shards/rgb-00000.tar",
        "rgb_member_path": f"rgb/{stem}.png",
        "events_path": f"{root}/events.h5",
        "labels_path": f"{root}/labels.parquet",
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, list[dict]]:
    data_root = tmp_path / "dataset"
    media = data_root / "data" / "train.parquet"
    media.parent.mkdir(parents=True)
    rows = [row("train_a:0001"), row("train_b:0002"), row("select_a:0003")]
    pq.write_table(pa.Table.from_pylist(rows), media)
    candidate = {
        "schema": "eap_detection_sequence_split_v1",
        "dataset_revision": REVISION,
        "media_sha256": sha256(media),
        "source_split": "train",
        "seed": 7,
        "assignments": {
            "train": ["train_a", "train_b"],
            "selection": ["select_a"],
            "calibration": ["calibrate_a"],
        },
    }
    candidate_path = tmp_path / "candidate.json"
    write_json(candidate_path, candidate)
    manifest = {
        "schema": "eap_rgb_pilot_v1",
        "dataset_revision": REVISION,
        "source_split": "train",
        "seed": 7,
        "source_media_sha256": sha256(media),
        "candidate_split_sha256": sha256(candidate_path),
        "train": {
            "sequences": ["train_a", "train_b"],
            "sample_tokens": ["train_a:0001", "train_b:0002"],
        },
        "selection": {"sequences": ["select_a"], "sample_tokens": ["select_a:0003"]},
        "overfit16": {"sample_tokens": ["train_a:0001"]},
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, candidate_path, data_root, manifest, rows


def rewrite_media(data_root: Path, rows: list[dict], candidate_path: Path, manifest: dict) -> None:
    media = data_root / "data" / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), media)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["media_sha256"] = sha256(media)
    write_json(candidate_path, candidate)
    manifest["source_media_sha256"] = sha256(media)
    manifest["candidate_split_sha256"] = sha256(candidate_path)


def test_valid_admission_returns_bounded_receipt(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, _, _ = fixture(tmp_path)
    receipt = validate_pilot_admission(manifest_path, candidate_path, data_root)
    assert receipt["status"] == "PASS"
    assert receipt["canonical_rows_verified"] == 3
    assert receipt["train_token_count"] == 2
    assert receipt["selection_token_count"] == 1
    assert receipt["calibration_sequences_selected"] == 0


def test_rejects_train_selection_token_leakage(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, _ = fixture(tmp_path)
    manifest["selection"]["sample_tokens"] = ["train_a:0001"]
    manifest["selection"]["sequences"] = ["train_a"]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="candidate selection|overlap"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_substituted_media_even_with_same_tokens(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, _, rows = fixture(tmp_path)
    altered = copy.deepcopy(rows)
    altered[0]["rgb_member_path"] = "rgb/replaced.png"
    pq.write_table(pa.Table.from_pylist(altered), data_root / "data" / "train.parquet")
    with pytest.raises(ValueError, match="Canonical data/train.parquet differs"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_sequence_assigned_to_wrong_role(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, _ = fixture(tmp_path)
    manifest["selection"] = {
        "sequences": ["train_b"],
        "sample_tokens": ["train_b:0002"],
    }
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="candidate selection"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_test_split_metadata(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, rows = fixture(tmp_path)
    altered = copy.deepcopy(rows)
    altered[2]["split"] = "test"
    rewrite_media(data_root, altered, candidate_path, manifest)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Canonical media identity differs"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_duplicate_tokens_before_parquet_access(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, _ = fixture(tmp_path)
    manifest["train"]["sample_tokens"].append("train_a:0001")
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_calibration_sequence_selected_for_training(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, rows = fixture(tmp_path)
    manifest["train"] = {
        "sequences": ["calibrate_a"],
        "sample_tokens": ["calibrate_a:0004"],
    }
    manifest["overfit16"] = {"sample_tokens": ["calibrate_a:0004"]}
    rows.append(row("calibrate_a:0004"))
    rewrite_media(data_root, rows, candidate_path, manifest)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="candidate train|calibration"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)


def test_rejects_path_escaping_public_train_sequence(tmp_path: Path) -> None:
    manifest_path, candidate_path, data_root, manifest, rows = fixture(tmp_path)
    altered = copy.deepcopy(rows)
    altered[0]["events_path"] = "data/test/train_a/events.h5"
    rewrite_media(data_root, altered, candidate_path, manifest)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not belong to public train"):
        validate_pilot_admission(manifest_path, candidate_path, data_root)
