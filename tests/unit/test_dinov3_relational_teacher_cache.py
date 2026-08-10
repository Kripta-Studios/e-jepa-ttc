"""Tests for DINOv3 Relational Teacher cache wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
    track_ids = np.array(["trk_A", "trk_B", "trk_A"]) # Two tracks, one object appears twice? No, two objects, one appears twice. Or just two objects in same frame. 
    sequence_ids = np.array(["seq1", "seq1", "seq2"])
    common_square = np.array([[0,0,10,10], [5,5,15,15], [0,0,10,10]], dtype=np.float32)
    relations = np.zeros((3, 2, 6, 32, 32), dtype=np.float16)
    valid = np.ones((3, 2, 6, 32, 32), dtype=np.uint8)
    
    np.savez_compressed(
        shard_path,
        sample_tokens=tokens,
        track_ids=track_ids,
        sequence_ids=sequence_ids,
        common_square_xyxy=common_square,
        relation_targets=relations,
        relation_valid=valid,
    )
    
    import hashlib
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
        },
        "teacher": {
            "model_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
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
    from e_jepa_ttc.artifacts.hashing import sign_artifact
    sign_artifact(manifest_payload)
    
    manifest_path.write_text(json.dumps(manifest_payload))
    return manifest_path


def test_dino_cache_wrapper_success(dummy_teacher_cache: Path) -> None:
    records = [
        {"sample_token": "t1", "track_id": "trk_A", "sequence_id": "seq1", "event_v4_common_square_xyxy": [0,0,10,10]},
        {"sample_token": "t2", "track_id": "trk_B", "sequence_id": "seq1", "event_v4_common_square_xyxy": [5,5,15,15]},
        {"sample_token": "t3", "track_id": "trk_A", "sequence_id": "seq2", "event_v4_common_square_xyxy": [0,0,10,10]},
    ]
    base_dataset = DummyDataset(records)
    
    import hashlib
    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()
    
    import json
    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]
    
    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset, # type: ignore
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
        {"sample_token": "t1", "track_id": "trk_WRONG", "sequence_id": "seq1", "event_v4_common_square_xyxy": [0,0,10,10]},
        {"sample_token": "t2", "track_id": "trk_B", "sequence_id": "seq1", "event_v4_common_square_xyxy": [5,5,15,15]},
        {"sample_token": "t3", "track_id": "trk_A", "sequence_id": "seq2", "event_v4_common_square_xyxy": [0,0,10,10]},
    ]
    base_dataset = DummyDataset(records)
    
    import hashlib
    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()
    
    import json
    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]
    
    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset, # type: ignore
        manifest_path=dummy_teacher_cache,
        expected_artifact_sha256=expected_artifact_sha,
        expected_manifest_sha256=manifest_sha,
    )
    
    with pytest.raises(ValueError, match="track_id mismatch"):
        _ = wrapper[0]


def test_dino_cache_wrapper_crop_mismatch(dummy_teacher_cache: Path) -> None:
    records = [
        # Match token and track, but mismatch crop
        {"sample_token": "t1", "track_id": "trk_A", "sequence_id": "seq1", "event_v4_common_square_xyxy": [0,0,10,11]},
        {"sample_token": "t2", "track_id": "trk_B", "sequence_id": "seq1", "event_v4_common_square_xyxy": [5,5,15,15]},
        {"sample_token": "t3", "track_id": "trk_A", "sequence_id": "seq2", "event_v4_common_square_xyxy": [0,0,10,10]},
    ]
    base_dataset = DummyDataset(records)
    
    import hashlib
    manifest_sha = hashlib.sha256(dummy_teacher_cache.read_bytes()).hexdigest()
    
    import json
    signed_manifest = json.loads(dummy_teacher_cache.read_text())
    expected_artifact_sha = signed_manifest["artifact_sha256"]
    
    wrapper = DINOv3RelationalTeacherDataset(
        base_dataset, # type: ignore
        manifest_path=dummy_teacher_cache,
        expected_artifact_sha256=expected_artifact_sha,
        expected_manifest_sha256=manifest_sha,
    )
    
    with pytest.raises(ValueError, match="common crop mismatch"):
        _ = wrapper[0]
