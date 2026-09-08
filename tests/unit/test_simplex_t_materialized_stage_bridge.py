"""Bridge invocation tests with stub children: no model execution or optimizer."""

import json
import runpy
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


@pytest.mark.parametrize(
    "mode", ["verify", "resume", "missing", "resource", "negative", "bad_pin", "no_latent"]
)
def test_stage_bridge_preserves_outcomes_and_exact_resume(tmp_path, monkeypatch, mode):
    work = Path(__file__).resolve().parents[2]
    root = work / "artifacts/simplex_t/T0" / ("bridge_fixture_" + uuid.uuid4().hex)
    (root / "launches").mkdir(parents=True)
    stage = "T3" if mode == "negative" else "T4" if mode == "no_latent" else "T2"
    template = root / "launches" / f"{stage}.json"
    if mode != "no_latent":
        template.write_text("{}", encoding="utf-8")
    if mode in {"negative", "no_latent"}:
        (root / "launches" / f"{stage}.decision.json").write_text(
            json.dumps(
                {
                    "schema": "simplex_t_technical_launch_decision_v1"
                    if mode == "no_latent"
                    else "simplex_t_practical_launch_decision_v1",
                    "stage": stage,
                    "eligible": False,
                    "optimizer_updates": 0,
                    "scientific_completion": False,
                }
            ),
            encoding="utf-8",
        )
    freeze_launch = tmp_path / "freeze_launch.json"
    freeze_launch.write_text("{}", encoding="utf-8")
    names = {
        "materializer": "materialize_simplex_t_technical_phases.py",
        "attempt": "execute_simplex_t_phase_attempt.py",
        "worker": "execute_simplex_t_frozen_phase.py",
    }
    args = [
        "bridge",
        "--freeze-launch",
        str(freeze_launch),
        "--freeze-launch-sha256",
        sha256(freeze_launch),
        "--campaign-root",
        str(root),
        "--stage",
        stage,
        "--other-reserved-bytes",
        "0",
        "--own-reserved-bytes",
        "8388608",
    ]
    for flag, name in names.items():
        args.extend(
            [
                f"--{flag}-sha256",
                "0" * 64
                if mode == "bad_pin" and flag == "worker"
                else sha256(work / "scripts" / name),
            ]
        )
    if mode == "resume":
        args.extend(["--resume", "--report", str(root / "attempt.json")])
    else:
        args.append("--verify-only")
    monkeypatch.setattr(sys, "argv", args)
    module = runpy.run_path(str(work / "scripts/run_simplex_t_materialized_stage.py"))
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
        assert ("--report" in command) == (mode == "resume")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module["subprocess"], "run", child)
    if mode == "bad_pin":
        with pytest.raises(ValueError, match="changed"):
            module["main"]()
        assert not calls
    else:
        assert module["main"]() == {"missing": 10, "resource": 3}.get(mode, 0)
        assert len(calls) == (1 if mode in {"missing", "resource", "negative", "no_latent"} else 2)
