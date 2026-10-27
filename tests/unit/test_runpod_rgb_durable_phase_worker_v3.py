"""V3 binds uploader V6 and keeps eviction payloads off argv."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import runpod_rgb_durable_phase_worker_v3 as wrapper


def test_v2_config_resolves_v6_entrypoints() -> None:
    module = wrapper.install_v3()
    root = Path(__file__).resolve().parents[2]
    config = module.load_config(
        root / "artifacts/runpod_migration_20261026/RGB_DURABLE_PHASE_WORKER_CONFIG_V2.json"
    )
    for key in ("r4_refiner", "r2_generation"):
        argv = config["uploads"][key]["argv"]
        assert any(item.endswith("upload_rgb_sparse_frames_v6.py") for item in argv)
        assert not any(item.endswith("upload_rgb_sparse_frames_v5.py") for item in argv)


def test_eviction_payload_uses_stdin_not_argv(tmp_path: Path) -> None:
    module = wrapper.install_v3()
    authority = tmp_path / "authority.json"
    authority.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0)

    instance = object.__new__(module.Worker)
    instance.config = {
        "eviction": {
            "authority_path": str(authority),
            "remote_root": "/workspace/ejepa_rgb_frames",
            "ssh_argv_prefix": ["ssh", "host"],
        }
    }
    instance.run = run
    item = {
        "sparse_tar_bytes": 3,
        "sparse_tar_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
    }
    instance._upload = lambda key: (
        ({"manifest_sha256": "c" * 64}, {"data/train/a/rgb_shards/a.tar": item})
        if key == "r4_refiner"
        else ({}, {"data/train/b/rgb_shards/b.tar": item})
    )
    module.validate_eviction_authority = lambda path: {}
    module.sha256 = lambda path: "d" * 64
    instance.evict()
    assert "records" not in str(captured["argv"])
    payload = json.loads(str(captured["input"]))
    assert payload["records"][0]["relative"] == "data/train/a/rgb_shards/a.tar"
    assert captured["text"] is True


def test_psutil_identity_preserves_accented_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [r"C:\Users\Álvaro Schwiedop\python.exe", "worker.py"]
    process = SimpleNamespace(
        is_running=lambda: True,
        status=lambda: wrapper.psutil.STATUS_RUNNING,
        cmdline=lambda: list(argv),
    )
    monkeypatch.setattr(wrapper.psutil, "Process", lambda pid: process)
    assert wrapper.process_matches(3352, argv)
