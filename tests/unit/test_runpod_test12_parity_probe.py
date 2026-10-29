"""Fail-closed provenance checks for the isolated Test12 parity probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "test12_probe", Path(__file__).resolve().parents[2] / "scripts/runpod_test12_parity_probe.py"
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_rejects_changed_source_even_with_correct_relocation(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("original", encoding="utf-8")
    frozen = {"sources": {r"C:\old\checkout\src\module.py": probe.digest(source)}}
    assert "src/module.py" in probe.validate_sources(tmp_path, frozen)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen"):
        probe.validate_sources(tmp_path, frozen)


def test_rejects_changed_frozen_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"value": 1}), encoding="utf-8")
    original_sha = probe.digest(manifest)
    manifest.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen identity mismatch"):
        probe.read_bound(manifest, original_sha)


def test_wrong_raw_size_rejected_before_output_or_preprocessing(tmp_path: Path) -> None:
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"partial")
    output = tmp_path / "output"
    args = argparse.Namespace(raw_root=tmp_path, output=output)
    plan = {"raw_relative_path": "raw.h5", "raw_bytes": 1024}
    with pytest.raises(ValueError, match="Raw file size differs"):
        probe.execute(args, plan, {})
    assert not output.exists()


def test_platform_normalization_preserves_members_and_compression(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("events.npy")
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, b"payload" * 100)
    before = path.read_bytes()
    assert probe.normalize_npz_platform(path) == 1
    after = path.read_bytes()
    assert len(before) == len(after)
    assert sum(a != b for a, b in zip(before, after, strict=True)) == 1
    with zipfile.ZipFile(path) as archive:
        assert archive.read("events.npy") == b"payload" * 100
        assert archive.getinfo("events.npy").create_system == 0
    assert probe.normalize_npz_platform(path) == 0
