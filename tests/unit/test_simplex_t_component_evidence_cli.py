"""Admission failures must precede producer/model evidence reads."""

import importlib.util
import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import component_verification


@pytest.mark.parametrize("failure", ["profile", "schema", "output", "reservation"])
def test_component_cli_rejects_before_evidence_loading(tmp_path, monkeypatch, failure):
    script = Path(__file__).resolve().parents[2] / "scripts/verify_simplex_t_component_evidence.py"
    spec = importlib.util.spec_from_file_location("component_evidence_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / "work"
    work.mkdir()
    local = work / "local.json"
    local.write_text(json.dumps({"worktree": str(work)}), encoding="utf-8")
    profile = work / "profile.json"
    profile.write_text(
        json.dumps(
            {"schema": "bad" if failure == "schema" else "simplex_t_component_evidence_profile_v1"}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.json" if failure == "output" else work / "out.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI configuration reached evidence loading")

    monkeypatch.setattr(component_verification, "verify_acknowledged_producers", forbidden)
    monkeypatch.setattr(module.torch, "set_num_threads", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify",
            "--local-paths",
            str(local),
            "--profile",
            str(profile),
            "--profile-sha256",
            "0" * 64 if failure == "profile" else sha256(profile),
            "--other-reserved-bytes",
            "-1" if failure == "reservation" else "0",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        module.main()
    assert not output.exists()
