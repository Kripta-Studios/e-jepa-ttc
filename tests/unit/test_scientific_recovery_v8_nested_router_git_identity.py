from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "run_scientific_recovery_v8_nested_router.py"
    spec = importlib.util.spec_from_file_location("v8_nested_router_git_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inner(commit: str, dirty: bool | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_sha256": "a" * 64,
        "git_commit": commit,
        "status": "completed",
    }
    if dirty is not None:
        payload["git_dirty"] = dirty
    return payload


def test_merged_inner_oof_rejects_missing_git_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "require_clean_scientific_worktree",
        lambda: {"git_commit": "head", "git_dirty": False},
    )
    sources = [_inner("abc", False), _inner("abc", False), _inner("abc", None)]
    with pytest.raises(module.RouterStageError, match="omitted git_dirty"):
        module._merged_inner_oof_git_identity(sources)


def test_merged_inner_oof_rejects_dirty_or_disagreeing_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "require_clean_scientific_worktree",
        lambda: {"git_commit": "head", "git_dirty": False},
    )
    dirty = [_inner("abc", False), _inner("abc", True), _inner("abc", False)]
    with pytest.raises(module.RouterStageError, match="produced dirty"):
        module._merged_inner_oof_git_identity(dirty)
    disagree = [_inner("abc", False), _inner("def", False), _inner("abc", False)]
    with pytest.raises(module.RouterStageError, match="disagree on git_commit"):
        module._merged_inner_oof_git_identity(disagree)


def test_merged_inner_oof_records_observed_clean_head(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "require_clean_scientific_worktree",
        lambda: {"git_commit": "merge-head", "git_dirty": False},
    )
    sources = [_inner("train-head", False) for _ in range(3)]
    identity = module._merged_inner_oof_git_identity(sources)
    assert identity == {"git_commit": "merge-head", "git_dirty": False}
