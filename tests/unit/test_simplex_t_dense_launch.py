"""DENSE launch publication requires inspection; no replay or optimizer calls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("fault", ["none", "missing", "resource", "authority"])
def test_dense_launch_waits_for_all_catalogs_and_inspection(tmp_path, monkeypatch, fault):
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_simplex_t_dense_launch.py"
    spec = importlib.util.spec_from_file_location("dense_launch_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / script.name))
    temporal = tmp_path / "artifacts/simplex_t/T1"
    for outer in range(2 if fault == "missing" else 3):
        path = temporal / f"compiled_context/outer{outer}/COMPILED.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"outer": outer}))
    index = temporal / "dense_query_context_index/INDEX_MANIFEST.json"
    index.parent.mkdir()
    index.write_text(json.dumps({"queries": 21471}))
    monkeypatch.setattr(
        module, "POOL_INDEX_SHA256", {"DENSE_OLD": hashlib.sha256(index.read_bytes()).hexdigest()}
    )
    dedup = temporal / "dense_query_context_dedup/DEDUP_MANIFEST.json"
    dedup.parent.mkdir()
    dedup.write_text("{}")
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path)}))
    output = tmp_path / "artifacts/launch.json"
    argv = [
        "runner",
        "--local-paths",
        str(local),
        "--output",
        str(output),
        "--other-reserved-bytes",
        "34000000000",
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": fault != "resource",
            "written_volume_free_bytes": [100_000_000_000],
        },
    )
    calls = []

    def inspect(local, candidate, digest, **kwargs):
        calls.append(True)
        assert kwargs["inspect_only"] is True
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest
        config = json.loads(candidate.read_text(encoding="utf-8"))
        assert set(config["d0_reuse"]) == {"0", "1", "2"}
        assert config["reserved_output_bytes"] == 8443789312
        if fault == "authority":
            raise ValueError("wrong owner time scope")
        return {"status": "EXPANDED_LAUNCH_INSPECTED_NOT_REPLAY_OR_SCIENTIFIC_ADMISSION"}

    monkeypatch.setattr(module, "run_configured_expanded_replay", inspect)
    if fault == "authority":
        with pytest.raises(ValueError, match="owner time"):
            module.main()
    else:
        assert module.main() == (10 if fault == "missing" else 3 if fault == "resource" else 0)
    if fault != "none":
        assert not output.exists()
        assert len(calls) == (1 if fault == "authority" else 0)
    else:
        before = (output.read_bytes(), output.stat().st_mtime_ns)
        argv.append("--verify-only")
        assert module.main() == 0
        assert (output.read_bytes(), output.stat().st_mtime_ns) == before
        assert len(calls) == 2
