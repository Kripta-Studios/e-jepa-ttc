"""Native PowerShell shell checks with help-only verifiers, never campaign work."""

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode",
    ["validate", "verified", "bad_hash", "verifier_report", "duplicate_report", "wrong_root"],
)
def test_campaign_shell_pins_and_verifier_skip(tmp_path, mode):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is required for native shell integration")
    work = Path(__file__).resolve().parents[2]
    script = work / "scripts/run_simplex_t_companion.py"
    command = dict(
        script="scripts/run_simplex_t_companion.py",
        sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        arguments=["--help"],
    )
    if mode in {"verifier_report", "duplicate_report"}:
        command["attempt_report"] = True
    if mode == "duplicate_report":
        command["arguments"] = ["--report=existing.json"]
    plan = tmp_path / "plan.json"
    spec = dict(
        schema="simplex_t_orchestration_plan_v1",
        steps=[dict(id="help_fixture", run=command, resume=command, verify=command)],
    )
    if mode == "wrong_root":
        spec["required_run_root"] = str(work / "artifacts/simplex_t/T0/not_the_logs")
    plan.write_text(json.dumps(spec), encoding="utf-8")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    output = work / "artifacts/simplex_t/T0" / ("orchestrator_fixture_" + uuid.uuid4().hex)
    args = [
        pwsh,
        "-NoProfile",
        "-File",
        str(work / "scripts/Invoke-SimplexTCampaign.ps1"),
        "-Plan",
        str(plan),
        "-PlanSha256",
        "0" * 64 if mode == "bad_hash" else digest,
        "-PythonExecutable",
        sys.executable,
        "-RunRoot",
        str(output),
    ]
    if mode == "validate":
        args.append("-ValidateOnly")
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=90)
    if mode in {"bad_hash", "verifier_report", "duplicate_report", "wrong_root"}:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        if mode == "validate":
            assert not output.exists()
        else:
            assert (output / "PLAN_SHA256.txt").read_text() == digest
            assert len(list(output.glob("*.verify.log"))) == 1
            assert not list(output.glob("*.run.log"))
            assert not (output / "ORCHESTRATOR.lock").exists()
            state = json.loads((output / "CAMPAIGN_STATE.json").read_text())
            assert state["status"] == "ALL_DECLARED_STEPS_VERIFIED"
            assert state["scientific_completion"] is False
            assert state["verified_steps_this_invocation"] == ["help_fixture"]
            assert len(list(output.glob("*.exit.json"))) == 1
            # A killed orchestrator can leave an unlocked file. There is no
            # matching live fixture worker; auto-resume must reverify outputs.
            (output / "ORCHESTRATOR.lock").touch()
            again = subprocess.run(
                args, capture_output=True, text=True, encoding="utf-8", timeout=90
            )
            assert again.returncode == 0, again.stdout + again.stderr
            assert len(list(output.glob("*.verify.log"))) == 2
            assert not list(output.glob("*.run.log"))
            events = [
                json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
            ]
            assert len({row["run_id"] for row in events}) == 2
            assert any(row["status"] == "RECOVERED_TERMINAL_ORCHESTRATOR" for row in events)
            assert not (output / "ORCHESTRATOR.lock").exists()


@pytest.mark.parametrize("attempt_report", [False, True])
def test_resource_retry_uses_resume_and_keeps_logs(tmp_path, attempt_report):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is required for native shell integration")
    work = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex
    output = work / "artifacts/simplex_t/T0" / ("orchestrator_retry_fixture_" + token)
    state = output / "fixture_attempts.json"
    script = work / "scripts" / (".orchestrator_fixture_" + token + ".py")
    # A uniquely owned temporary native entry point, removed after this test.
    # It performs no model work and changes only its fixture state.
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[2]); rows=json.loads(p.read_text()) if p.exists() else []\n"
        "if sys.argv[1]=='verify': sys.exit(0 if len(rows)==2 else 10)\n"
        "rows.append(sys.argv[1]); p.write_text(json.dumps(rows))\n"
        "if len(sys.argv)>3:\n"
        " assert sys.argv[3]=='--report'\n"
        " with Path(sys.argv[4]).open('x') as f: json.dump(rows,f)\n"
        "sys.exit(3 if len(rows)==1 else 0)\n",
        encoding="utf-8",
    )
    try:
        digest = hashlib.sha256(script.read_bytes()).hexdigest()

        def command(mode):
            result = dict(
                script=script.relative_to(work).as_posix(),
                sha256=digest,
                arguments=[mode, str(state)],
            )
            if attempt_report and mode != "verify":
                result["attempt_report"] = True
            return result

        plan = tmp_path / "retry.json"
        plan.write_text(
            json.dumps(
                dict(
                    schema="simplex_t_orchestration_plan_v1",
                    steps=[
                        dict(
                            id="resource_fixture",
                            run=command("run"),
                            resume=command("resume"),
                            verify=command("verify"),
                        )
                    ],
                )
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                pwsh,
                "-NoProfile",
                "-File",
                str(work / "scripts/Invoke-SimplexTCampaign.ps1"),
                "-Plan",
                str(plan),
                "-PlanSha256",
                hashlib.sha256(plan.read_bytes()).hexdigest(),
                "-PythonExecutable",
                sys.executable,
                "-RunRoot",
                str(output),
                "-ResourceRetrySeconds",
                "1",
                "-MaxResourceRetries",
                "3",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(state.read_text()) == ["run", "resume"]
        assert len(list(output.glob("*.run.log"))) == 2
        assert len(list(output.glob("*.verify.log"))) == 3
        assert not (output / "ORCHESTRATOR.lock").exists()
        invocations = list(output.glob("*.command.json"))
        assert len(invocations) == 5
        state_value = json.loads((output / "CAMPAIGN_STATE.json").read_text())
        assert state_value["status"] == "ALL_DECLARED_STEPS_VERIFIED"
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert any(row["status"] == "PAUSED_RESOURCE" for row in events)
        assert len(list(output.glob("*.exit.json"))) == 5
        reports = list(output.glob("*.run.log.report.json"))
        assert len(reports) == (2 if attempt_report else 0)
        if attempt_report:
            assert sorted(len(json.loads(p.read_text())) for p in reports) == [1, 2]
    finally:
        script.unlink()


@pytest.mark.parametrize(
    "code,script,message,expected",
    [
        (
            1,
            "build_simplex_t_context_features.py",
            "RuntimeError: RESOURCE_PAUSE at completed query boundary",
            3,
        ),
        (
            1,
            "build_simplex_t_context_features.py",
            "RuntimeError: RESOURCE_PAUSE before model load",
            3,
        ),
        (1, "another.py", "RuntimeError: RESOURCE_PAUSE before model load", 1),
        (1, "build_simplex_t_context_features.py", "ValueError: invalid producer", 1),
        (
            1,
            "build_simplex_t_context_features.py",
            "RuntimeError: RESOURCE_PAUSE before model load\nValueError: later failure",
            1,
        ),
        (
            0,
            "build_simplex_t_context_features.py",
            "RuntimeError: RESOURCE_PAUSE before model load",
            0,
        ),
    ],
)
def test_exact_legacy_resource_pause_adapter(tmp_path, code, script, message, expected):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell required")
    source = Path(__file__).resolve().parents[2] / "scripts/Invoke-SimplexTCampaign.ps1"
    log = tmp_path / "worker.log"
    log.write_text(message + "\n", encoding="utf-8")

    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    command = (
        "$t=$null; $e=$null; $ast=[System.Management.Automation.Language.Parser]::ParseFile("
        + quote(source)
        + ",[ref]$t,[ref]$e); "
        "$node=$ast.Find({param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] "
        "-and $n.Name -eq 'Resolve-WorkerExitCode'},$true); "
        ". ([ScriptBlock]::Create($node.Extent.Text)); "
        f"Resolve-WorkerExitCode -Code {code} -ScriptPath {quote(script)} -LogPath {quote(log)}"
    )
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-EncodedCommand",
            base64.b64encode(command.encode("utf-16-le")).decode("ascii"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == expected
