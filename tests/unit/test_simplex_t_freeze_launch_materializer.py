"""An absent prerequisite never produces a nominal scientific freeze launch."""

import runpy
import sys
import uuid
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


@pytest.mark.parametrize("mode", ["verify", "run", "bad_pin", "same_output"])
def test_missing_inputs_are_explicit_and_do_not_write(tmp_path, monkeypatch, capsys, mode):
    work = Path(__file__).resolve().parents[2]
    root = work / "artifacts/simplex_t/T0" / ("freeze_cli_fixture_" + uuid.uuid4().hex)
    local = tmp_path / "local.json"
    local.write_text("{}", encoding="utf-8")
    output = root / "launch.json"
    argv = [
        "materialize",
        "--local-paths",
        str(local),
        "--local-paths-sha256",
        "0" * 64 if mode == "bad_pin" else sha256(local),
        "--source-configuration",
        str(root / "sources.json"),
        "--evidence-profile",
        str(root / "evidence.json"),
        "--preparation",
        str(root / "preparation.json"),
        "--code-commit",
        "a" * 40,
        "--output",
        str(output),
        "--freeze-output",
        str(output if mode == "same_output" else root / "freeze.json"),
        "--other-reserved-bytes",
        "0",
    ]
    if mode == "verify":
        argv.append("--verify-only")
    monkeypatch.setattr(sys, "argv", argv)
    module = runpy.run_path(str(work / "scripts/materialize_simplex_t_freeze_launch.py"))
    if mode == "bad_pin":
        with pytest.raises(ValueError, match="changed"):
            module["main"]()
    elif mode == "same_output":
        with pytest.raises(SystemExit):
            module["main"]()
    else:
        assert module["main"]() == (10 if mode == "verify" else 2)
        assert "WAITING_EXACT_FREEZE_INPUTS" in capsys.readouterr().out
    assert not root.exists()
