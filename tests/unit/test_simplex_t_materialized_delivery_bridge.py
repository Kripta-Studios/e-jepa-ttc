"""Metadata success must still invoke the actual delivery verifier or worker."""

import runpy
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


@pytest.mark.parametrize(
    "mode", ["verify", "resume", "missing", "resource", "bad_pin", "worker_error"]
)
def test_delivery_bridge_preserves_worker_status(tmp_path, monkeypatch, mode):
    work = Path(__file__).resolve().parents[2]
    root = work / "artifacts/simplex_t/T0" / ("delivery_bridge_fixture_" + uuid.uuid4().hex)
    (root / "launches").mkdir(parents=True)
    template = root / "launches/T6.json"
    template.write_text("{}", encoding="utf-8")
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text("{}", encoding="utf-8")
    names = {
        "materializer": "materialize_simplex_t_delivery.py",
        "attempt": "execute_simplex_t_postprocessing_attempt.py",
        "worker": "postprocess_simplex_t_campaign.py",
    }
    argv = [
        "bridge",
        "--campaign-root",
        str(root),
        "--run-root",
        str(root / "logs"),
        "--reconciliation",
        str(reconciliation),
        "--reconciliation-sha256",
        sha256(reconciliation),
        "--analysis-commit",
        "a" * 40,
        "--other-reserved-bytes",
        "0",
        "--own-reserved-bytes",
        "8388608",
    ]
    for flag, name in names.items():
        argv.extend(
            [
                f"--{flag}-sha256",
                "0" * 64
                if mode == "bad_pin" and flag == "worker"
                else sha256(work / "scripts" / name),
            ]
        )
    argv.append("--resume" if mode == "resume" else "--verify-only")
    monkeypatch.setattr(sys, "argv", argv)
    module = runpy.run_path(str(work / "scripts/run_simplex_t_materialized_delivery.py"))
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == work
        if len(calls) == 1:
            assert Path(command[2]).name == names["materializer"]
            return SimpleNamespace(returncode={"missing": 10, "resource": 3}.get(mode, 0))
        assert Path(command[2]).name == names["attempt"]
        assert command[command.index("--template-sha256") + 1] == sha256(template)
        assert ("--resume" in command) == (mode == "resume")
        assert ("--verify-only" in command) == (mode != "resume")
        return SimpleNamespace(returncode=2 if mode == "worker_error" else 0)

    monkeypatch.setattr(module["subprocess"], "run", child)
    if mode == "bad_pin":
        with pytest.raises(ValueError, match="changed"):
            module["main"]()
        assert not calls
    else:
        assert module["main"]() == {"missing": 10, "resource": 3, "worker_error": 2}.get(mode, 0)
        assert len(calls) == (1 if mode in {"missing", "resource"} else 2)
