"""Transfer integrity and unchanged-recipe tests; no cloud or GPU required."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = load("runpod_release_transfer")
BUILD = load("prepare_runpod_release_bundle")
CLOUD = load("runpod_release_cloud")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/outside", "a\\..\\outside"])
def test_destination_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        TRANSFER.safe_path(tmp_path, name)


def test_registered_windows_roots_only():
    assert BUILD.relative(Path("E:/data/a.bin"), {"E:/data": "data"}) == "data/a.bin"
    with pytest.raises(ValueError):
        BUILD.relative(Path("D:/secret/key"), {"E:/": "data"})


def test_stage_profiles_keep_event_roles_separate():
    row = {"kind": "public", "modality": "events", "role": "calibration"}
    assert not TRANSFER.selected(row, "rgb")
    assert not TRANSFER.selected(row, "events-train")
    assert TRANSFER.selected(row, "events-calibration")
    private = {"kind": "private", "profiles": ["phase-inputs", "all"]}
    assert not TRANSFER.selected(private, "rgb")
    assert TRANSFER.selected(private, "phase-inputs")


def test_range_resume_requires_correct_server_ack(tmp_path, monkeypatch):
    data = b"abcdef"
    row = {
        "target": "data/a",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": "https://example.invalid/file",
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.part").write_bytes(b"abc")
    response = io.BytesIO(b"def")
    response.status = 206
    response.headers = {"Content-Range": "bytes 3-5/6"}
    monkeypatch.setattr(TRANSFER.urllib.request, "urlopen", lambda *a, **kw: response)
    TRANSFER.fetch_one(tmp_path, row)
    assert (tmp_path / "data/a").read_bytes() == data
    (tmp_path / "data/a").unlink()
    (tmp_path / "data/a.part").write_bytes(b"abc")
    response = io.BytesIO(b"abc")
    response.status = 206
    response.headers = {"Content-Range": "bytes 0-2/6"}
    with pytest.raises(ValueError, match="range"):
        TRANSFER.fetch_one(tmp_path, row)
    assert (tmp_path / "data/a.part").read_bytes() == b"abc"


def test_unpack_hash_and_population(tmp_path):
    incoming, root = tmp_path / "incoming", tmp_path / "root"
    incoming.mkdir()
    data = b"preserved bytes"
    row = {
        "target": "workspace/source",
        "kind": "private",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    archive = incoming / "private_000.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(row["target"])
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
    parts = [
        {"name": archive.name, "bytes": archive.stat().st_size, "sha256": TRANSFER.digest(archive)}
    ]
    (incoming / "PARTS.json").write_text(json.dumps(parts))
    TRANSFER.unpack(root, incoming, {"files": [row]})
    assert (root / row["target"]).read_bytes() == data
    (root / row["target"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="altered"):
        TRANSFER.unpack(root, incoming, {"files": [row]})


def test_cloud_config_keeps_scientific_recipe_and_original_unchanged(tmp_path):
    original = {
        "seed": 7,
        "experiment_id": "old",
        "data": {"root": "E:/eAP_dataset", "manifest": "fixed", "expected_manifest_sha256": "pin"},
        "model": {"portable_root": "old", "queries": 50},
        "optimization": {"batch_size": 4, "precision": "fp32", "learning_rate": 0.0002},
    }
    cloud = CLOUD.relocate_config(original, tmp_path, "r2", 13)
    assert cloud["optimization"] == original["optimization"]
    assert cloud["data"]["expected_manifest_sha256"] == "pin"
    assert cloud["model"]["queries"] == 50
    assert cloud["seed"] == 13
    assert original["seed"] == 7
    assert original["data"]["root"] == "E:/eAP_dataset"
