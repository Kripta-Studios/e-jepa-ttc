"""Source wiring fixtures; no payloads, targets or optimizers are opened."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def entry():
    script = (
        Path(__file__).resolve().parents[2] / "scripts/prepare_simplex_t_source_configuration.py"
    )
    spec = importlib.util.spec_from_file_location("source_builder_fixture", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("amended", [False, True])
def test_build_keeps_all_pools_and_fold_bindings(entry, tmp_path, monkeypatch, amended):
    pools = {
        "EXPANSION_POOL_PLAN.json": (
            "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849"
        ),
        "MATCHED_CONTROL_POOL_PLAN.json": (
            "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c"
        ),
    }
    selection = {"path": str(tmp_path / "selection.json"), "sha256": "b" * 64}
    if amended:
        pools["MATCHED_CONTROL_POOL_DENSITY_20260908.json"] = (
            "00322ff06093aa5dc46c2aaed26e42961dc3349fc85823001672d0c593cca122"
        )
        config = tmp_path / "configs/experiment/simplex_t_throughput_amendment.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"id": "SIMPLEX_T_THROUGHPUT_2026-09-08"}))
        plan = tmp_path / "artifacts/simplex_t/T0/MATCHED_CONTROL_POOL_DENSITY_20260908.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            json.dumps({"dense_membership_unchanged": True, "query_selection": selection})
        )
    monkeypatch.setattr(entry, "sha256", lambda path: pools.get(path.name, "b" * 64))
    for prefix, pool in (("expansion", "D1"), ("dense", "DENSE_OLD")):
        for outer in range(3):
            folder = tmp_path / f"artifacts/simplex_t/T1/compiled_{prefix}_context/outer{outer}"
            folder.mkdir(parents=True)
            (folder / "COMPILED.json").write_text(
                json.dumps(
                    {
                        "outer": outer,
                        "pool": pool,
                        "cache_identity_sha256": str(outer) * 64,
                        **({"query_selection": selection} if amended else {}),
                    }
                )
            )
    ack = {
        "interfaces": {"role_manifest": {"roles": {"original": ["old"], "expansion": ["new"]}}},
        "producers": {"authoritative_historical_manifest": {"sha256": "a" * 64}},
    }
    result = entry.configuration(tmp_path, ack)
    assert set(result) == {
        "schema",
        "original_sequences",
        "expansion_sequences",
        "original",
        "expansion",
        "dense",
        "matched",
    }
    for key in ("expansion", "dense"):
        assert set(result[key]) == {"0", "1", "2"}
        for outer in range(3):
            row = result[key][str(outer)]
            assert row["cache_identity_sha256"] == str(outer) * 64
            assert row["metadata"]["root"] == row["labels"]["root"] == "garl"
    assert result["matched"]["pool_sha256"] == result["dense"]["0"]["pool_sha256"]
    if amended:
        assert result["matched"]["pool"]["relative_path"].endswith(
            "MATCHED_CONTROL_POOL_DENSITY_20260908.json"
        )


def test_missing_cache_returns_exact_dependencies_without_writes(
    entry, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(entry, "__file__", str(tmp_path / "scripts/builder.py"))
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path), "shared_coordination": str(tmp_path)}))
    output = tmp_path / "artifacts/sources.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "builder",
            "--local-paths",
            str(local),
            "--output",
            str(output),
            "--other-reserved-bytes",
            "0",
            "--verify-only",
        ],
    )
    monkeypatch.setattr(
        entry,
        "admitted",
        lambda *a: {"has_headroom": True, "written_volume_free_bytes": [100_000_000_000]},
    )
    monkeypatch.setattr(entry, "verified_ack", lambda *a: {})
    monkeypatch.setattr(entry, "verify_expansion_authority", lambda *a, **k: {})
    assert entry.main() == 10
    assert len(json.loads(capsys.readouterr().out)["missing"]) == 9
    assert not output.parent.exists()
