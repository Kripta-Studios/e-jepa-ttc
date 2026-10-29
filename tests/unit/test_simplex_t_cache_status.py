"""Operational status must not confuse a receipt with live execution or full QA."""

import json
from types import SimpleNamespace

from e_jepa_ttc.simplex_t.cache_status import context_cache_status


def test_empty_cache_is_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr("psutil.process_iter", lambda fields: [])
    result = context_cache_status(tmp_path)
    assert result["status"] == "NOT_STARTED"
    assert result["scientific_completion_proven"] is False


def test_receipt_without_process_is_resumable_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr("psutil.process_iter", lambda fields: [])
    cache = tmp_path / "artifacts/simplex_t/T1/context_features_fp32"
    cache.mkdir(parents=True)
    (cache / "family00_query00000.json").write_text(json.dumps({"rows": 16}))
    (cache / "family00_query00000.npz").write_bytes(b"not a verified payload")
    result = context_cache_status(tmp_path)
    assert result["status"] == "RESUMABLE_PARTIAL_EXTRACTION"
    assert result["declared_observations"] == 16
    assert result["payload_hashes_rechecked_by_status"] is False
    assert result["last_saved_content_audit"] is None


def test_live_execution_requires_matching_command_and_worktree(tmp_path, monkeypatch):
    process = SimpleNamespace(
        info={
            "pid": 123,
            "name": "python.exe",
            "cwd": str(tmp_path),
            "cmdline": ["python", "scripts/build_simplex_t_context_features.py"],
        }
    )
    monkeypatch.setattr("psutil.process_iter", lambda fields: [process])
    result = context_cache_status(tmp_path)
    assert result["status"] == "LIVE_CONTEXT_EXTRACTION"
    process.info["cwd"] = str(tmp_path.parent)
    assert context_cache_status(tmp_path)["status"] == "NOT_STARTED"
