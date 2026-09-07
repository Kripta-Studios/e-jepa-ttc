"""Native shell -> real attempt controller -> fixture worker, with zero fit updates."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_shell_retries_partial_postprocessing_and_verifies_delivery(tmp_path):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("native PowerShell required")
    repository = Path(__file__).resolve().parents[2]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shell = scripts / "Invoke-SimplexTCampaign.ps1"
    shutil.copyfile(repository / "scripts" / shell.name, shell)
    controller = repository / "scripts/execute_simplex_t_postprocessing_attempt.py"
    entry = scripts / "fixture_entry.py"
    # Load the real controller unchanged, replacing only host resource admission.
    # The scientific worker is a fixture; this does not verify a scientific graph.
    entry.write_text(
        "import importlib.util,sys\n"
        f"sys.path.insert(0,{str(repository / 'src')!r})\n"
        f"s=importlib.util.spec_from_file_location('real_attempt',{str(controller)!r})\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        f"m.__file__={str(scripts / controller.name)!r}\n"
        "m.admitted=lambda *a:{'has_headroom':True,'written_volume_free_bytes':[100000000000]}\n"
        "sys.exit(m.main())\n",
        encoding="utf-8",
    )
    worker = scripts / "postprocess_simplex_t_campaign.py"
    worker.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--launch')+1]);c=json.loads(p.read_text(encoding='utf-8'))\n"
        "delivery=Path(c['delivery']['output']); marker=delivery/'DELIVERY.json'\n"
        "if '--verify-only' in sys.argv: sys.exit(0 if marker.exists() else 10)\n"
        "analysis=Path(c['output']);analysis.mkdir()\n"
        "(analysis/'PARTIAL.txt').write_text('preserve fixture')\n"
        "prior=list(p.parent.parent.glob('attempt_*/analysis/PARTIAL.txt'))\n"
        "if len(prior)==1: sys.exit(3)\n"
        "delivery.mkdir();marker.write_text('{}');sys.exit(0)\n",
        encoding="utf-8",
    )
    root = tmp_path / "artifacts/attempts"
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "schema": "simplex_t_postprocessing_launch_v3",
                "roots": {"work": str(tmp_path)},
                "output": str(root / "analysis"),
                "delivery": {"output": str(root / "delivery"), "analysis_commit": "a" * 40},
                "freeze_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "--template",
        str(template),
        "--template-sha256",
        hashlib.sha256(template.read_bytes()).hexdigest(),
        "--worker-sha256",
        hashlib.sha256(worker.read_bytes()).hexdigest(),
        "--attempt-root",
        str(root),
        "--other-reserved-bytes",
        "0",
        "--own-reserved-bytes",
        "100000000",
    ]

    def command(extra):
        return {
            "script": entry.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
            "arguments": arguments + extra,
        }

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "simplex_t_orchestration_plan_v1",
                "steps": [
                    {
                        "id": "postprocessing_fixture",
                        "run": command([]),
                        "resume": command(["--resume"]),
                        "verify": command(["--verify-only"]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    logs = tmp_path / "artifacts/logs"
    invocation = [
        pwsh,
        "-NoProfile",
        "-File",
        str(shell),
        "-Plan",
        str(plan),
        "-PlanSha256",
        hashlib.sha256(plan.read_bytes()).hexdigest(),
        "-PythonExecutable",
        sys.executable,
        "-RunRoot",
        str(logs),
        "-ResourceRetrySeconds",
        "1",
        "-MaxResourceRetries",
        "3",
    ]
    completed = subprocess.run(
        invocation, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(list(root.glob("attempt_*/analysis/PARTIAL.txt"))) == 2
    assert len(list(root.glob("attempt_*/delivery/DELIVERY.json"))) == 1
    assert len(list(logs.glob("*.run.log"))) == 2
    assert not (logs / "ORCHESTRATOR.lock").exists()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }
    resumed = subprocess.run(
        invocation + ["-Resume"], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list(logs.glob("*.run.log"))) == 2
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    } == before
