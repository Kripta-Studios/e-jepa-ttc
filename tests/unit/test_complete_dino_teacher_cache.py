from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (
    CompleteDinoTeacherCache,
    write_complete_mmap_cache,
)
from e_jepa_ttc.distillation.dinov3_relational import A4_RELATION_OFFSETS
from e_jepa_ttc.scientific_provenance import ScientificProvenanceError


def _base_manifest() -> dict[str, object]:
    payload = {
        "artifact_type": "scientific_recovery_v8_complete_dino_teacher_v1",
        "status": "completed",
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "ttc_labels_read": False,
            "row_count": 2,
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
            "endpoint_metadata_stored_in_complete_cache": True,
            "raw_rgb_sha256_stored_per_endpoint": True,
        },
        "code_identity": {"git_commit": "abc123", "git_dirty": False},
        "relations": {
            "type": "local_cosine",
            "offsets_dy_dx": [list(offset) for offset in A4_RELATION_OFFSETS],
            "grid_height": 32,
            "grid_width": 32,
        },
    }
    sign_artifact(payload)
    return payload


def _write_cache(tmp_path: Path, *, tokens: list[str] | None = None) -> Path:
    tokens = tokens or ["tok-a", "tok-b"]
    shape = (len(tokens), 2, len(A4_RELATION_OFFSETS), 32, 32)
    write_complete_mmap_cache(
        tmp_path,
        tokens=tokens,
        track_ids=["trk-a", "trk-b"][: len(tokens)],
        sequence_ids=["seq-a", "seq-b"][: len(tokens)],
        relation_targets=np.zeros(shape, dtype=np.float16),
        relation_valid=np.ones(shape, dtype=np.uint8),
        crops=np.zeros((len(tokens), 4), dtype=np.float32),
        manifest=_base_manifest(),
    )
    return tmp_path / "manifest.json"


def _open(manifest: Path, *, allowed: set[str] | None = None) -> CompleteDinoTeacherCache:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return CompleteDinoTeacherCache.open_verified(
        manifest,
        expected_artifact_sha256=str(payload["artifact_sha256"]),
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        allowed_sample_tokens=allowed,
    )


def test_complete_cache_open_verified_roundtrip(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = _open(manifest)
    assert payload.tokens() == frozenset({"tok-a", "tok-b"})
    track, sequence, relations, valid, crop = payload["tok-a"]
    assert track == "trk-a"
    assert sequence == "seq-a"
    assert relations.shape == (2, len(A4_RELATION_OFFSETS), 32, 32)
    assert valid.shape == relations.shape
    assert crop.shape == (4,)


def test_complete_cache_missing_token_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        _open(manifest, allowed={"tok-a", "missing-token"})


def test_complete_cache_corruption_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    (tmp_path / "teacher.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        _open(manifest)


def test_complete_cache_stale_identity_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    with pytest.raises(ValueError, match="artifact identity"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256="0" * 64,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_partial_coverage_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["coverage"] = 0.5
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_dirty_materialization_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["code_identity"] = {"git_commit": "abc123", "git_dirty": True}
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_extra_token_fails(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["rows"].append(
        {
            "token_id": "tok-extra",
            "track_id": "trk-x",
            "sequence_id": "seq-x",
            "row_index": 2,
        }
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    payload["row_count_observed"] = 3
    payload["unexpected"] = 1
    sign_artifact(payload)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected tokens"):
        CompleteDinoTeacherCache.open_verified(
            manifest,
            expected_artifact_sha256=str(payload["artifact_sha256"]),
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_complete_cache_bypass_env_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_cache(tmp_path)
    monkeypatch.setenv("DINO_ALLOW_PARTIAL_CACHE", "1")
    with pytest.raises(ScientificProvenanceError, match="DINO_ALLOW_PARTIAL_CACHE"):
        _open(manifest)


def test_training_modules_do_not_invoke_dino_materializer() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "materialize_dinov3_relational_teacher",
        "transformers.AutoModel",
        "DINO_NUM_CHUNKS",
    )
    for relative in (
        "src/e_jepa_ttc/training/causal_scale_eap.py",
        "src/e_jepa_ttc/training/scientific_recovery_v8_trainer.py",
        "src/e_jepa_ttc/training/scientific_recovery_v8_jobs.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
