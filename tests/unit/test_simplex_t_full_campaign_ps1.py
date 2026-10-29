"""Native full wrapper wiring with fixture producers; no scientific fit or QA claim."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_preparation_to_science_transition_and_repeat(tmp_path):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("native PowerShell required")
    repository = Path(__file__).resolve().parents[2]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("Invoke-SimplexTCampaign.ps1", "Invoke-SimplexTFullCampaign.ps1"):
        shutil.copyfile(repository / "scripts" / name, scripts / name)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    freeze = tmp_path / "artifacts/freeze_launch.json"
    worker = scripts / "materialize_simplex_t_freeze_launch.py"
    worker.write_text(
        "import sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--output')+1])\n"
        "if '--verify-only' in sys.argv: sys.exit(0 if p.exists() else 10)\n"
        "p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}');sys.exit(0)\n",
        encoding="utf-8",
    )
    delivery = scripts / "fixture_delivery.py"
    marker = tmp_path / "artifacts/delivery.json"
    delivery.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"p=Path({str(marker)!r})\n"
        "if '--verify-only' in sys.argv: sys.exit(0 if p.exists() else 10)\n"
        "p.write_text('{}');sys.exit(0)\n",
        encoding="utf-8",
    )
    builder = scripts / "prepare_simplex_t_scientific_plan.py"
    builder.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--output')+1])\n"
        "root=sys.argv[sys.argv.index('--run-root')+1]\n"
        f"c=dict(script='scripts/fixture_delivery.py',sha256={digest(delivery)!r},arguments=[])\n"
        "v={**c,'arguments':['--verify-only']}\n"
        "plan=dict(schema='simplex_t_orchestration_plan_v1',required_run_root=root,"
        "steps=[dict(id='fixture_T6',run=c,resume=c,verify=v)])\n"
        "if p.exists(): assert json.loads(p.read_text())==plan\n"
        "else: p.write_text(json.dumps(plan))\n",
        encoding="utf-8",
    )
    command = dict(
        script="scripts/materialize_simplex_t_freeze_launch.py",
        sha256=digest(worker),
        arguments=["--output", str(freeze)],
    )
    prep = tmp_path / "prepare.json"
    prep.write_text(
        json.dumps(
            dict(
                schema="simplex_t_orchestration_plan_v1",
                steps=[
                    dict(
                        id="freeze_inputs",
                        run=command,
                        resume=command,
                        verify={**command, "arguments": [*command["arguments"], "--verify-only"]},
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text("{}", encoding="utf-8")
    logs = tmp_path / "artifacts/logs"
    config = tmp_path / "full.json"
    config.write_text(
        json.dumps(
            dict(
                schema="simplex_t_full_orchestration_v1",
                run_root=str(logs),
                campaign_root=str(tmp_path / "artifacts/science"),
                freeze_launch=str(freeze),
                code_commit="a" * 40,
                preparation_plan=dict(path=str(prep), sha256=digest(prep)),
                reconciliation=dict(path=str(reconciliation), sha256=digest(reconciliation)),
                orchestrator_sha256=digest(scripts / "Invoke-SimplexTCampaign.ps1"),
                scientific_plan_builder_sha256=digest(builder),
                other_reserved_bytes=0,
                own_reserved_bytes=8_388_608,
            )
        ),
        encoding="utf-8",
    )
    args = [
        shell,
        "-NoProfile",
        "-File",
        str(scripts / "Invoke-SimplexTFullCampaign.ps1"),
        "-Configuration",
        str(config),
        "-ConfigurationSha256",
        digest(config),
        "-PythonExecutable",
        sys.executable,
        "-MaxResourceRetries",
        "1",
    ]
    for _ in range(2):
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    assert freeze.is_file() and marker.is_file()
    assert len(list((logs / "preparation_logs").glob("*.run.log"))) == 1
    assert len(list((logs / "scientific_logs").glob("*.run.log"))) == 1
    assert len(list(logs.glob("plan_generation_*.log"))) == 2
    assert (
        json.loads((logs / "scientific_logs/CAMPAIGN_STATE.json").read_text())["status"]
        == "ALL_DECLARED_STEPS_VERIFIED"
    )
