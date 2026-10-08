from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operational.sota_eval import reuse


def _sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _row(query_id: str, anchor: int = 1_000, box: int = 1) -> dict:
    row = {
        "query_id": query_id,
        "sequence_id": "sequence",
        "anchor_us": anchor,
        "windows_us": [[700, 800], [800, 900], [900, 1_000]],
        "boxes_xyxy3": [[box, 2, 3, 4]] * 3,
        "square_xyxy": [1, 2, 3, 4],
        "bbox_sources": [{"frame_index": 1, "timestamp_us": 900, "sha256": "a"}],
        "raw_path": "raw.hdf5",
        "raw_stat": {"size": 1, "mtime_ns": 2},
        "calibration_sha256": "b",
        "scenario_family": "CCRs",
        "speed_bucket": "low",
        "target_type": "car",
    }
    row["metadata_sha256"] = _sha(row)
    return row


def test_row_index_matches_only_complete_query_independent_metadata() -> None:
    old = _row("old:00")
    new = _row("new:17")
    assert set(reuse._index_rows([old], "old")) == set(reuse._index_rows([new], "new"))


def test_row_tamper_is_rejected() -> None:
    row = _row("old:00")
    row["boxes_xyxy3"][0][0] = 99
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        reuse._index_rows([row], "tampered")


def test_duplicate_logical_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate logical query"):
        reuse._index_rows([_row("a"), _row("b")], "duplicate")


def test_same_anchor_wrong_box_is_collision() -> None:
    with pytest.raises(ValueError, match="identity collision"):
        reuse._index_rows([_row("a", box=1), _row("b", box=9)], "collision")


def test_immutable_fragment_rejects_existing_difference(tmp_path: Path) -> None:
    path = tmp_path / "fragment.json"
    reuse._write_immutable_json(path, {"status": "REUSED", "value": 1})
    reuse._write_immutable_json(path, {"status": "REUSED", "value": 1})
    with pytest.raises(ValueError, match="immutable reuse output differs"):
        reuse._write_immutable_json(path, {"status": "REUSED", "value": 2})


def test_reuse_rejects_either_nested_root(tmp_path: Path) -> None:
    parent = tmp_path / "source"
    child = parent / "expanded"
    parent.mkdir()
    child.mkdir()
    with pytest.raises(ValueError, match="separate"):
        reuse.reuse_predictions(parent, child)
    with pytest.raises(ValueError, match="separate"):
        reuse.reuse_predictions(child, parent)


def test_source_verification_detects_fragment_tamper(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    predictions = root / "predictions"
    predictions.mkdir(parents=True)
    row = _row("sequence:00")
    manifest = {"rows": [row]}
    manifest_path = root / "QUERY_MANIFEST.json"
    reuse.atomic_json(manifest_path, manifest)
    freeze = {"manifest_sha256": reuse.digest(manifest_path), "sources": {}, "models": {}}
    freeze_path = root / "INFERENCE_FREEZE.json"
    reuse.atomic_json(freeze_path, freeze)
    fragment = {
        "query_id": row["query_id"],
        "sequence_id": row["sequence_id"],
        "anchor_us": row["anchor_us"],
        "binding_sha256": reuse.digest(freeze_path),
        "optimizer_updates": 0,
    }
    fragment_path = predictions / "query_00000.json"
    reuse.atomic_json(fragment_path, fragment)
    fragment_sha = reuse.digest(fragment_path)
    fragment_path.with_suffix(".sha256").write_text(fragment_sha + "\n", encoding="ascii")
    reuse.atomic_json(
        root / "PREDICTIONS_SEALED.json",
        {
            "status": "COMPLETE",
            "queries": 1,
            "manifest_sha256": reuse.digest(manifest_path),
            "binding_sha256": reuse.digest(freeze_path),
            "fragments": {"predictions/query_00000.json": fragment_sha},
            "optimizer_updates": 0,
        },
    )
    monkeypatch.setattr(reuse, "_verify_model_binding", lambda _: None)
    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: iter(()))
    fragment_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fragment checksum mismatch"):
        reuse._verify_source_run(root)
