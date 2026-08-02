from __future__ import annotations

import json

import pytest

from scripts.execute_garl_release_cache_matrix import _require_readiness


def test_cache_matrix_requires_readiness_for_unbounded_runs(tmp_path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"long_training_authorized": False}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="blocked by readiness gates"):
        _require_readiness(path, bounded=False)


def test_cache_matrix_allows_bounded_smoke_before_long_readiness(tmp_path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"long_training_authorized": False}), encoding="utf-8")

    assert _require_readiness(path, bounded=True)["long_training_authorized"] is False
