"""Native PowerShell shell checks with help-only verifiers, never campaign work."""

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["validate", "verified", "bad_hash"])
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
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                schema="simplex_t_orchestration_plan_v1",
                steps=[dict(id="help_fixture", run=command, resume=command, verify=command)],
            )
        ),
        encoding="utf-8",
    )
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
    if mode == "bad_hash":
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


def test_resource_retry_uses_resume_and_keeps_logs(tmp_path):
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
        "sys.exit(3 if len(rows)==1 else 0)\n",
        encoding="utf-8",
    )
    try:
        digest = hashlib.sha256(script.read_bytes()).hexdigest()

        def command(mode):
            return dict(
                script=script.relative_to(work).as_posix(),
                sha256=digest,
                arguments=[mode, str(state)],
            )

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
    finally:
        script.unlink()
