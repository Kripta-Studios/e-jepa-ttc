"""Launch composition cannot substitute arbitrary QA or practical gate callbacks."""

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import configured_execution as module


@pytest.mark.parametrize("mode", ["ok", "qa", "changed", "outside", "stage", "pause"])
def test_configured_launch(tmp_path: Path, monkeypatch, mode: str):
    local = tmp_path / "paths.json"
    local.write_text(json.dumps({"worktree": str(tmp_path), "shared_coordination": str(tmp_path)}))
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    calls = []
    source = object()

    def admission(*args, **kwargs):
        calls.append("admission")
        if mode == "qa":
            raise ValueError("missing real H16 evidence")

    def open_sources(*args):
        calls.append("open")
        return source, {}

    def run(*args, **kwargs):
        calls.append("run")
        assert kwargs["sources"] is source
        assert kwargs["validate_stage_gate"] is None
        assert kwargs["publications"] == {}
        if mode == "changed":
            freeze.write_text("changed")
        kwargs["validate_authority_and_qa"]()
        return {"status": "PAUSED_RESOURCE" if mode == "pause" else "published"}

    monkeypatch.setattr(module, "validate_scientific_admission", admission)
    monkeypatch.setattr(module, "open_acknowledged_source_configuration", open_sources)
    monkeypatch.setattr(module, "run_historical_cohort_phase", run)
    kwargs = dict(
        local_paths=local,
        source_configuration=local,
        source_configuration_sha256=sha256(local),
        evidence_profile=local,
        evidence_profile_sha256=sha256(local),
        freeze=freeze,
        freeze_sha256=sha256(freeze),
        roots={"work": tmp_path},
        execution=tmp_path.parent if mode == "outside" else tmp_path / "execution",
        publication=tmp_path / "publication",
        stage="T6" if mode == "stage" else "T2",
        availability={},
        publications={},
        resource_ok=lambda: True,
        resume=False,
    )
    if mode in {"ok", "pause"}:
        result = module.execute_configured_phase(**kwargs)
        assert result["status"] == ("PAUSED_RESOURCE" if mode == "pause" else "published")
        assert calls == ["admission", "open", "run", "admission"]
    else:
        with pytest.raises(ValueError):
            module.execute_configured_phase(**kwargs)
        if mode in {"outside", "stage"}:
            assert not calls
        elif mode == "qa":
            assert calls == ["admission"]
        else:
            assert calls == ["admission", "open", "run"]
    assert not (tmp_path / "execution").exists()
