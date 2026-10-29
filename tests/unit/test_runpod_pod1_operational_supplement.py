from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.runpod_pod1_operational_supplement_v1 import collect, publish


def test_source_tree_includes_code_docs_and_excludes_data(tmp_path: Path) -> None:
    tree = tmp_path / "workspace"
    (tree / "src").mkdir(parents=True)
    (tree / "docs").mkdir()
    (tree / "artifacts").mkdir()
    (tree / "src/model.py").write_text("x = 1\n", encoding="utf-8")
    (tree / "docs/method.md").write_text("method\n", encoding="utf-8")
    (tree / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tree / "artifacts/checkpoint.pt").write_bytes(b"checkpoint")
    records = collect([], [], [f"bundle={tree}"])
    assert set(records) == {
        "bundle/docs/method.md",
        "bundle/pyproject.toml",
        "bundle/src/model.py",
    }


def test_supplement_tar_is_regular_and_primary_bound(tmp_path: Path) -> None:
    evidence = tmp_path / "status.json"
    evidence.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    records = collect([f"operations/status.json={evidence}"], [], [])
    output = tmp_path / "supplement.tar"
    receipt = publish(records, output, "a" * 64)
    assert receipt["status"] == "READY"
    with tarfile.open(output) as stream:
        assert all(member.isfile() for member in stream.getmembers())
        manifest = json.load(stream.extractfile("MANIFEST.json"))  # type: ignore[arg-type]
    assert manifest["primary_archive_sha256"] == "a" * 64
    assert manifest["global_atomic_snapshot_claimed"] is False


def test_binding_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "status.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SAFE/ARCHIVE"):
        collect([f"../escape={source}"], [], [])
