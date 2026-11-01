import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_evttc_transfer.py"


def test_missing_resource_has_exit_three(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(SCRIPT),
            "score",
            "--predictions",
            str(tmp_path / "absent.json"),
            "--targets",
            str(tmp_path / "targets.json"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert not (tmp_path / "out/report.json").exists()


def test_scoring_rejects_official_claim_and_preserves_existing_receipt(tmp_path: Path) -> None:
    files = {"predictions": [], "targets": {"rows": []}, "policy": {"scope": "equivalent_official"}}
    for key, value in files.items():
        (tmp_path / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
    command = [sys.executable, "-X", "utf8", str(SCRIPT), "score"]
    for key in files:
        command += [f"--{key}", str(tmp_path / f"{key}.json")]
    command += ["--output", str(tmp_path / "out")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    (tmp_path / "policy.json").write_text('{"scope":"diagnostic_partial"}', encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    receipt = tmp_path / "out/report.json"
    before = receipt.read_bytes()
    receipt.write_text('{"old_evidence":true}', encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert receipt.read_text(encoding="utf-8") == '{"old_evidence":true}'
    assert json.loads(before)["counts"]["requested"] == 0
