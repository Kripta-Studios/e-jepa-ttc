"""Coverage never claims payload integrity or scientific completion."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.cache_completion import complete_d0_receipts


@pytest.mark.parametrize("mode", ["absent", "partial", "resource"])
def test_read_only_cli_reports_incomplete_without_creating_files(tmp_path, monkeypatch, mode):
    script = Path(__file__).resolve().parents[2] / "scripts/audit_simplex_t_context_cache.py"
    spec = importlib.util.spec_from_file_location("cache_cli_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache, index = tmp_path / "cache", tmp_path / "index"
    if mode == "partial":
        cache.mkdir()
        index.mkdir()
        array = index / "query_context_index.npz"
        np.savez(
            array, producer_family=np.repeat(np.array([[0], [4], [8]], np.int64), 8192, axis=1)
        )
        (cache / "IDENTITY.json").write_text(
            json.dumps({"index_sha256": module.compute_file_hash(str(array))})
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--cache",
            str(cache),
            "--index",
            str(index),
            "--dedup",
            str(tmp_path / "dedup"),
            "--verify-only",
            "--other-reserved-bytes",
            "22000000000",
        ],
    )
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": mode != "resource",
            "written_volume_free_bytes": [100_000_000_000],
        },
    )
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert module.main() == (3 if mode == "resource" else 10)
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("fault", ["none", "missing", "alias", "unbacked", "family", "resource"])
def test_d0_coverage_requires_exact_query_family_pairs(tmp_path, fault):
    assignments = np.array([[0, 3], [4, 7], [8, 11]], dtype=np.int64)
    for row in assignments:
        for query, family in enumerate(row):
            if fault == "missing" and family == 11:
                continue
            name = f"family{family:02d}_query{query:05d}"
            if fault == "alias" and family == 11:
                name += "_alias"
            path = tmp_path / (name + ".json")
            path.write_text(json.dumps({"query": query, "family": int(family)}))
            if fault != "unbacked" or family != 11:
                path.with_suffix(".npz").write_bytes(b"not validated by coverage")
    if fault == "family":
        assignments[0, 0] = 4
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    if fault in {"alias", "unbacked", "family", "resource"}:
        with pytest.raises((ValueError, InterruptedError)):
            complete_d0_receipts(tmp_path, assignments, resource_ok=lambda: fault != "resource")
    else:
        result = complete_d0_receipts(tmp_path, assignments, resource_ok=lambda: True)
        assert result is None if fault == "missing" else len(result) == 6
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == before
