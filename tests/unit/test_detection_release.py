from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.infer_detection_release import read_tokens, verify_release, write_json_bytes


def _release(root: Path) -> None:
    root.mkdir()
    (root / "weights.pt").write_bytes(b"checkpoint")
    write_json_bytes(
        root / "ranker.json",
        {"provenance": {"checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest()}},
    )
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    write_json_bytes(
        root / "MANIFEST.json",
        {
            "schema": "phase_detection_release_v1",
            "files": files,
            "checkpoint": "weights.pt",
            "ranker": "ranker.json",
        },
    )


def test_release_rejects_changed_checkpoint(tmp_path):
    root = tmp_path / "release"
    _release(root)
    verify_release(root)
    (root / "weights.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="hash"):
        verify_release(root)


def test_release_rejects_manifest_path_escape(tmp_path):
    root = tmp_path / "release"
    _release(root)
    value = json.loads((root / "MANIFEST.json").read_text())
    value["files"]["../outside"] = "0" * 64
    write_json_bytes(root / "MANIFEST.json", value)
    with pytest.raises(ValueError, match="path"):
        verify_release(root)


def test_release_binds_ranker_to_detector(tmp_path):
    root = tmp_path / "release"
    _release(root)
    write_json_bytes(root / "ranker.json", {"provenance": {"checkpoint_sha256": "0" * 64}})
    value = json.loads((root / "MANIFEST.json").read_text())
    value["files"]["ranker.json"] = hashlib.sha256((root / "ranker.json").read_bytes()).hexdigest()
    write_json_bytes(root / "MANIFEST.json", value)
    with pytest.raises(ValueError, match="ranker"):
        verify_release(root)


@pytest.mark.parametrize("payload", [{"sample_tokens": ["a"], "labels": []}, ["a", "a"], []])
def test_release_refuses_labels_duplicates_and_empty_population(tmp_path, payload):
    path = tmp_path / "tokens.json"
    write_json_bytes(path, payload)
    with pytest.raises(ValueError):
        read_tokens(path)


def test_json_hash_matches_persisted_bytes_on_windows(tmp_path):
    path = tmp_path / "predictions.json"
    digest = write_json_bytes(path, {"sample_tokens": ["á:1"], "value": 1.5})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert b"\r\n" not in path.read_bytes()
