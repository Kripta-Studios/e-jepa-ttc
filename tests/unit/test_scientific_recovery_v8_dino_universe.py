"""V8 DINO teacher must cover the frozen 8192-row event universe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from e_jepa_ttc.training.scientific_recovery_v8_trainer import (
    assert_v8_dino_teacher_matches_source_rows,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, row_count: int) -> tuple[dict[str, str], Path]:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"scope": {"row_count": row_count}}), encoding="utf-8")
    return {"manifest_sha256": _sha(path)}, path


def test_v8_dino_row_count_must_match_expected_source_rows(tmp_path: Path) -> None:
    dino, path = _manifest(tmp_path, row_count=8192)
    assert_v8_dino_teacher_matches_source_rows(
        dino, expected_source_train_rows=8192, manifest_path=path
    )


def test_v8_dino_2048_cache_is_refused_for_8192_source_rows(tmp_path: Path) -> None:
    dino, path = _manifest(tmp_path, row_count=2048)
    with pytest.raises(ValueError, match="row_count differs from expected_source_train_rows"):
        assert_v8_dino_teacher_matches_source_rows(
            dino, expected_source_train_rows=8192, manifest_path=path
        )


def test_v8_dino_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    _, path = _manifest(tmp_path, row_count=8192)
    with pytest.raises(ValueError, match="manifest file hash differs"):
        assert_v8_dino_teacher_matches_source_rows(
            {"manifest_sha256": "0" * 64},
            expected_source_train_rows=8192,
            manifest_path=path,
        )
