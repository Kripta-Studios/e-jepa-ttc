from __future__ import annotations

import argparse
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from e_jepa_ttc.detection.temporal_data import TemporalRGBIndex

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/prepare_detection_temporal_population.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_detection_temporal_population_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(token: str, start: int, end: int) -> dict[str, object]:
    return {
        "sample_token": token,
        "sequence_id": "OYgB6RGWcq",
        "rgb_exposure_start_timestamp_us": start,
        "rgb_exposure_end_timestamp_us": end,
    }


def test_unreadable_canonical_history_is_masked_without_alternative_backfill() -> None:
    module = _module()
    older = "OYgB6RGWcq:older"
    bad = module.KNOWN_BAD_TOKEN
    current = "OYgB6RGWcq:current"
    index = TemporalRGBIndex([_row(older, 0, 10), _row(bad, 100, 110), _row(current, 200, 210)])
    media = {
        older: {"status": "PASS", "sequence_id": "OYgB6RGWcq"},
        bad: {"status": "UNREADABLE", "sequence_id": "OYgB6RGWcq"},
    }
    entries, counts = module.build_temporal_entries(
        index=index,
        current_tokens=[current],
        role="train",
        sequence_owner={"OYgB6RGWcq": "train"},
        media_by_token=media,
        maximum_history_age_us=500_000,
    )
    entry = entries[0]
    assert entry["canonical_history_tokens_before_media"] == [older, bad]
    assert entry["history_tokens"] == [older, None]
    assert entry["history_mask"] == [True, False]
    assert entry["history_mask_reasons"] == [None, "known_unreadable_history"]
    assert entry["substitute_history_after_failure"] is False
    assert counts["masked_known_unreadable_history"] == 1


def test_maximum_age_mask_does_not_shift_an_older_candidate() -> None:
    module = _module()
    tokens = ["s:0", "s:1", "s:2", "s:3"]
    rows = [
        {
            "sample_token": token,
            "sequence_id": "s",
            "rgb_exposure_start_timestamp_us": index * 100,
            "rgb_exposure_end_timestamp_us": index * 100 + 10,
        }
        for index, token in enumerate(tokens)
    ]
    index = TemporalRGBIndex(rows)
    media = {token: {"status": "PASS", "sequence_id": "s"} for token in tokens[:-1]}
    entries, counts = module.build_temporal_entries(
        index=index,
        current_tokens=[tokens[-1]],
        role="train",
        sequence_owner={"s": "train"},
        media_by_token=media,
        maximum_history_age_us=150,
    )
    assert entries[0]["canonical_history_tokens_before_media"] == ["s:1", "s:2"]
    assert entries[0]["history_tokens"] == [None, "s:2"]
    assert counts["masked_maximum_age"] == 1


def test_new_history_member_is_strict_decoded_and_known_bad_remains_negative(
    tmp_path: Path,
) -> None:
    module = _module()
    relative = "data/train/s/rgb_shards/rgb-00000.tar"
    tar_path = tmp_path / relative
    tar_path.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    bad_raw = raw[:-30]
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("rgb/good.png")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
        bad_info = tarfile.TarInfo("rgb/bad.png")
        bad_info.size = len(bad_raw)
        archive.addfile(bad_info, io.BytesIO(bad_raw))
    tar_sha256 = module.sha256_file(tar_path)
    good = "s:good"
    bad = module.KNOWN_BAD_TOKEN
    metadata = {
        good: {
            "sequence_id": "s",
            "rgb_shard_path": relative,
            "rgb_member_path": "rgb/good.png",
        },
        bad: {
            "sequence_id": "OYgB6RGWcq",
            "rgb_shard_path": relative,
            "rgb_member_path": "rgb/bad.png",
        },
    }
    admitted, negatives, bindings = module.admit_history_media(
        data_root=tmp_path,
        metadata_by_token=metadata,
        history_tokens=[good, bad],
        current_media={},
        shard_receipts={
            relative: {
                "observed_sha256": tar_sha256,
                "observed_size_bytes": tar_path.stat().st_size,
            }
        },
        pinned_files={
            relative: {
                "sha256": tar_sha256,
                "size_bytes": tar_path.stat().st_size,
            }
        },
        known_bad_receipt={
            "sample_token": bad,
            "status": "FAIL",
            "error_type": "OSError",
            "error": "image file is truncated",
        },
    )
    assert admitted[good]["status"] == "PASS"
    assert admitted[good]["image_wh"] == [128, 128]
    assert admitted[bad]["status"] == "UNREADABLE"
    assert admitted[bad]["member_sha256"] == module.hashlib.sha256(bad_raw).hexdigest()
    assert admitted[bad]["tar_sha256"] == tar_sha256
    assert negatives == [admitted[bad]]
    assert bindings[relative]["status"] == "PASS_WITH_UNREADABLE_MEMBER"
    assert bindings[relative]["unreadable_history_members"] == 1


def test_run_masks_bound_unreadable_history_without_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    bad = module.KNOWN_BAD_TOKEN
    train_current = "OYgB6RGWcq:current"
    selection_current = "selection_sequence:current"
    relative = "data/train/OYgB6RGWcq/rgb_shards/rgb-00000.tar"
    member_path = "rgb/000000661_000000160.png"
    tar_path = tmp_path / relative
    tar_path.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), color=(1, 2, 3)).save(buffer, format="PNG")
    bad_raw = buffer.getvalue()[:-30]
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo(member_path)
        member.size = len(bad_raw)
        archive.addfile(member, io.BytesIO(bad_raw))
    tar_sha256 = module.sha256_file(tar_path)

    parquet_path = tmp_path / "data/train.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_token": bad,
            "split": "train",
            "sequence_id": "OYgB6RGWcq",
            "rgb_shard_path": relative,
            "rgb_member_path": member_path,
            "rgb_exposure_start_timestamp_us": 0,
            "rgb_exposure_end_timestamp_us": 10,
        },
        {
            "sample_token": train_current,
            "split": "train",
            "sequence_id": "OYgB6RGWcq",
            "rgb_shard_path": "unused-current-train.tar",
            "rgb_member_path": "rgb/current-train.png",
            "rgb_exposure_start_timestamp_us": 100,
            "rgb_exposure_end_timestamp_us": 110,
        },
        {
            "sample_token": selection_current,
            "split": "train",
            "sequence_id": "selection_sequence",
            "rgb_shard_path": "unused-current-selection.tar",
            "rgb_member_path": "rgb/current-selection.png",
            "rgb_exposure_start_timestamp_us": 0,
            "rgb_exposure_end_timestamp_us": 10,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    media_sha256 = module.sha256_file(parquet_path)

    def write(name: str, value: object) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    split = write(
        "split.json",
        {
            "schema": "eap_detection_sequence_split_v1",
            "media_sha256": media_sha256,
            "assignments": {
                "train": ["OYgB6RGWcq"],
                "selection": ["selection_sequence"],
                "calibration": ["calibration_sequence"],
            },
        },
    )
    full_selection = write(
        "full_selection.json",
        {
            "schema": "eap_detection_full_selection_v1",
            "sample_tokens": [selection_current],
        },
    )
    pinned = write(
        "pinned.json",
        {
            "schema_version": "eap_detection_public_files_v1",
            "files": {
                relative: {
                    "sha256": tar_sha256,
                    "size_bytes": tar_path.stat().st_size,
                }
            },
        },
    )
    failure_receipt = {
        "sample_token": bad,
        "sequence_id": "OYgB6RGWcq",
        "rgb_shard_path": relative,
        "rgb_member_path": member_path,
        "status": "FAIL",
        "error_type": "OSError",
        "error": "image file is truncated",
    }
    preflight = write("preflight.json", {"bad_tokens": [failure_receipt]})
    shard_ledger = tmp_path / "shards.jsonl"
    shard_ledger.write_text(
        json.dumps(
            {
                "rgb_shard_path": relative,
                "observed_sha256": tar_sha256,
                "observed_size_bytes": tar_path.stat().st_size,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = write(
        "manifest.json",
        {
            "schema": "eap_detection_campaign_manifest_v1",
            "candidate_split_sha256": module.sha256_file(split),
            "source_media_sha256": media_sha256,
            "roles": {"train": {"sample_tokens": [train_current]}},
        },
    )
    current_decision = {
        "member_sha256": "0" * 64,
        "member_size_bytes": 1,
    }
    media_admission = write(
        "media_admission.json",
        {
            "schema": "eap_detection_media_admission_v1",
            "source_pinned_inventory_sha256": module.sha256_file(pinned),
            "source_full_preflight_summary_sha256": module.sha256_file(preflight),
            "source_full_preflight_shard_ledger_sha256": module.sha256_file(shard_ledger),
            "final_media": {
                train_current: current_decision,
                selection_current: current_decision,
            },
        },
    )
    receipt = write(
        "receipt.json",
        {
            "schema": "eap_detection_readable_campaign_revision_v1",
            "new_manifest_sha256": module.sha256_file(manifest),
            "media_admission_sha256": module.sha256_file(media_admission),
            "full_selection_sha256": module.sha256_file(full_selection),
        },
    )

    monkeypatch.setattr(module, "EXPECTED_CURRENT_COUNT", 2)
    monkeypatch.setattr(module, "EXPECTED_TRAIN_CURRENT_COUNT", 1)
    monkeypatch.setattr(module, "EXPECTED_SELECTION_CURRENT_COUNT", 1)
    monkeypatch.setattr(module, "EXPECTED_UNIQUE_HISTORY_COUNT", 1)
    monkeypatch.setattr(module, "EXPECTED_NEW_HISTORY_COUNT", 1)
    monkeypatch.setattr(module, "EXPECTED_NEW_HISTORY_SHARD_COUNT", 1)
    monkeypatch.setattr(module, "EXPECTED_CANONICAL_SLOT_COUNTS", (0, 1))
    monkeypatch.setattr(
        module,
        "EXPECTED_ROLE_SUPPORT",
        {
            "train": {"history_0": 1, "history_1": 0, "history_2": 0},
            "selection": {"history_0": 1, "history_1": 0, "history_2": 0},
        },
    )
    output_dir = tmp_path / "output"
    summary = module.run(
        argparse.Namespace(
            data_root=tmp_path,
            split=split,
            v5_manifest=manifest,
            v5_receipt=receipt,
            v5_media_admission=media_admission,
            full_selection=full_selection,
            preflight_summary=preflight,
            preflight_shard_ledger=shard_ledger,
            pinned_manifest=pinned,
            output_dir=output_dir,
            maximum_history_age_us=500_000,
        )
    )
    admission = json.loads((output_dir / "temporal_media_admission.json").read_text())
    negative = json.loads((output_dir / "negative_audit.json").read_text())
    entry = admission["roles"]["train"]["entries"][0]
    assert summary["status"] == "PASS_WITH_EXPLICIT_HISTORY_MASKS"
    assert entry["canonical_history_tokens_before_media"] == [None, bad]
    assert entry["history_tokens"] == [None, None]
    assert entry["history_mask"] == [False, False]
    assert entry["history_mask_reasons"] == [
        "sequence_boundary",
        "known_unreadable_history",
    ]
    assert entry["substitute_history_after_failure"] is False
    assert negative["negative_count"] == 1
    assert negative["rows"][0]["member_sha256"] == module.hashlib.sha256(bad_raw).hexdigest()
    assert negative["rows"][0]["tar_sha256"] == tar_sha256
    assert negative["rows"][0]["error_type"] == "OSError"
