"""Tests for DINOv3 Relational Teacher cache wrapper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (
    DINOv3RelationalTeacherDataset,
)


class DummyDataset:
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx].copy()


@pytest.fixture
def dummy_teacher_cache(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"

    # Create dummy shard
    shard_path = tmp_path / "shard_0000.npz"
    tokens = np.array(["t1", "t2", "t3"])
    track_ids = np.array(["trk_A", "trk_B", "trk_A"])
    sequence_ids = np.array(["seq1", "seq1", "seq2"])
    common_square = np.array(
        [[0, 0, 10, 10], [5, 5, 15, 15], [0, 0, 10, 10]],
        dtype=np.float32,
    )
    relations = np.zeros((3, 2, 6, 32, 32), dtype=np.float16)
    valid = np.ones((3, 2, 6, 32, 32), dtype=np.uint8)
    rgb_sha256 = np.full((3, 2), "a" * 64, dtype="<U64")
    rgb_endpoint_indices = np.tile(np.array([0, 1], dtype=np.int32), (3, 1))
    rgb_frame_timestamps_us = np.array(
        [[1_000_000, 1_100_000], [2_000_000, 2_100_000], [3_000_000, 3_100_000]],
        dtype=np.int64,
    )
    event_windows_us = np.array(
        [
            [[950_000, 1_000_000], [1_050_000, 1_100_000]],
            [[1_950_000, 2_000_000], [2_050_000, 2_100_000]],
            [[2_950_000, 3_000_000], [3_050_000, 3_100_000]],
        ],
        dtype=np.int64,
    )
    rgb_shard_paths = np.full((3, 2), "train/seq/rgb.tar")
    rgb_member_paths = np.full((3, 2), "frame.jpg")

    np.savez_compressed(
        shard_path,
        sample_tokens=tokens,
        track_ids=track_ids,
        sequence_ids=sequence_ids,
        common_square_xyxy=common_square,
        relation_targets=relations,
        relation_valid=valid,
        rgb_sha256=rgb_sha256,
        rgb_endpoint_indices=rgb_endpoint_indices,
        rgb_frame_timestamps_us=rgb_frame_timestamps_us,
        event_windows_us=event_windows_us,
        rgb_shard_paths=rgb_shard_paths,
        rgb_member_paths=rgb_member_paths,
    )

    def _sha256_file(p: Path) -> str:
        d = hashlib.sha256()
        with p.open("rb") as f:
            d.update(f.read())
        return d.hexdigest()

    shard_sha = _sha256_file(shard_path)

    manifest_payload = {
        "artifact_sha256": "dummy_artifact_sha",
        "status": "passed",
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "ttc_labels_read": False,
            "row_count": 3,
            "endpoint_count_per_row": 2,
        },
        "claim_boundary": {
            "teacher_is_model_input": False,
            "validation_teacher_generation": False,
            "ttc_labels_read": False,
            "teacher_source_modality": "rgb",
            "event_tensor_used_as_teacher_input": False,
        },
        "teacher": {
            "model_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "source_modality": "rgb",
        },
        "source_rgb": {
            "endpoint_metadata_stored_in_shards": True,
            "raw_rgb_sha256_stored_per_endpoint": True,
        },
        "code_identity": {
            "git_commit": "abc123",
            "git_dirty": False,
        },
        "relations": {
            "type": "local_cosine",
            "offsets_dy_dx": [[0,1],[1,0],[0,2],[2,0],[1,1],[1,-1]],
            "grid_height": 32,
            "grid_width": 32,
        },
        "shards": [
            {
                "npz_path": "shard_0000.npz",
                "npz_sha256": shard_sha,
                "row_count": 3,
            }
        ]
    }

    # We need a valid artifact signature. We can just sign it.
    sign_artifact(manifest_payload)

    manifest_path.write_text(json.dumps(manifest_payload))
    return manifest_path


def _record(
    token: str, track_id: str, sequence_id: str, square: list[int]
) -> dict[str, object]:
    return {
        "sample_token": token,
        "track_id": track_id,
        "sequence_id": sequence_id,
        "event_v4_common_square_xyxy": square,
    }


def test_dino_cache_wrapper_success(dummy_teacher_cache: Path) -> None:
    records = [
        _record("t1", "trk_A", "seq1", [0, 0, 10, 10]),
        _record("t2", "trk_B", "seq1", [5, 5, 15, 15]),
        _record("t3", "trk_A", "seq2", [0, 0, 10, 10]),
    ]
    base_dataset = DummyDataset(records)

    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()

    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]

    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset,  # type: ignore
        manifest_path=dummy_teacher_cache,
        expected_artifact_sha256=expected_artifact_sha,
        expected_manifest_sha256=manifest_sha,
    )

    assert len(wrapper) == 3

    for i in range(3):
        out = wrapper[i]
        assert "dinov3_relation_targets" in out
        assert "dinov3_relation_valid" in out


def test_dino_cache_wrapper_track_id_mismatch(dummy_teacher_cache: Path) -> None:
    records = [
        # Match token, but mismatch track_id
        _record("t1", "trk_WRONG", "seq1", [0, 0, 10, 10]),
        _record("t2", "trk_B", "seq1", [5, 5, 15, 15]),
        _record("t3", "trk_A", "seq2", [0, 0, 10, 10]),
    ]
    base_dataset = DummyDataset(records)

    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()

    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]

    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset,  # type: ignore
        manifest_path=dummy_teacher_cache,
        expected_artifact_sha256=expected_artifact_sha,
        expected_manifest_sha256=manifest_sha,
    )

    with pytest.raises(ValueError, match="track_id mismatch"):
        _ = wrapper[0]


def test_dino_cache_wrapper_crop_mismatch(dummy_teacher_cache: Path) -> None:
    records = [
        # Match token and track, but mismatch crop
        _record("t1", "trk_A", "seq1", [0, 0, 10, 11]),
        _record("t2", "trk_B", "seq1", [5, 5, 15, 15]),
        _record("t3", "trk_A", "seq2", [0, 0, 10, 10]),
    ]
    base_dataset = DummyDataset(records)

    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()

    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]

    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset,  # type: ignore
        manifest_path=dummy_teacher_cache,
        expected_artifact_sha256=expected_artifact_sha,
        expected_manifest_sha256=manifest_sha,
    )

    with pytest.raises(ValueError, match="common crop mismatch"):
        _ = wrapper[0]


def test_dino_cache_wrapper_rejects_unproven_rgb_source(
    dummy_teacher_cache: Path,
) -> None:
    payload = json.loads(dummy_teacher_cache.read_text())
    payload["teacher"]["source_modality"] = "events"
    payload["claim_boundary"]["teacher_source_modality"] = "events"
    sign_artifact(payload)
    dummy_teacher_cache.write_text(json.dumps(payload))
    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()

    records = [
        _record("t1", "trk_A", "seq1", [0, 0, 10, 10]),
        _record("t2", "trk_B", "seq1", [5, 5, 15, 15]),
        _record("t3", "trk_A", "seq2", [0, 0, 10, 10]),
    ]
    with pytest.raises(ValueError, match="RGB source modality"):
        DINOv3RelationalTeacherDataset(
            DummyDataset(records),  # type: ignore[arg-type]
            manifest_path=dummy_teacher_cache,
            expected_artifact_sha256=payload["artifact_sha256"],
            expected_manifest_sha256=manifest_sha,
        )
